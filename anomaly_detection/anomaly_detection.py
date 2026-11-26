"""Detect anomalies using physical rules and leave-one-out GRU residuals."""

import argparse
import hashlib
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from src.models.gru.gru import (
    MEASUREMENT_COLUMNS,
    SensorGRU,
    TIME_STEP,
    build_sensor_arrays,
    normalize_measurements,
)


PROCESSED_DIR = ROOT / "data/processed"
INPUT_PATH = PROCESSED_DIR / "anomaly_period.csv"
CHECKPOINT_PATH = ROOT / "src/models/gru/checkpoints/best_model.pt"
OUTPUT_PATH = ROOT / "anomaly_detection/results/detections.csv"

# Default to a ten-minute slice of the later, unlabeled period.
START_TIME = 1728000
END_TIME = 1728600
CALIBRATE_ONLY = False

WINDOW_LENGTH = 40
BATCH_SIZE = 128
TIMESTAMPS_PER_CHUNK = 120

CALIBRATION_QUANTILE = 0.99
CALIBRATION_SAMPLE_SIZE = 4096
CALIBRATION_SEED = 42
# Calibrated on 4,096 readings per feature, validation times 1187880–1423530.
# Checkpoint SHA256: ebee071a63bdbf51bc4ced837fcd80822b14fb026a3292e04c9d49a2d61c7644
# Reproduce with --calibrate; thresholds are in native measurement units.
RESIDUAL_THRESHOLDS = {
    "temperature": 1.500306009521519,
    "humidity": 2.6222635302734463,
    "light": 284.90152954101563,
    "voltage": 0.14568523091125507,
}

KEY_COLUMNS = ["time", "nodeid"]
INPUT_COLUMNS = KEY_COLUMNS + MEASUREMENT_COLUMNS


def load_checkpoint():
    """Load the GRU and validate its normalization and target/neighbor ordering."""
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    model = SensorGRU()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    means = checkpoint["means"].numpy()
    stds = checkpoint["stds"].numpy()
    sensor_ids = checkpoint["sensor_ids"].numpy()
    neighbors = checkpoint["neighbors"].numpy()

    if not np.isfinite(means).all() or not np.isfinite(stds).all() or (stds <= 0).any():
        raise ValueError("Checkpoint normalization must have finite means and positive standard deviations")
    if neighbors.shape != (len(sensor_ids), 5):
        raise ValueError("Checkpoint must specify the target sensor and four neighbors")
    if (neighbors < 0).any() or (neighbors >= len(sensor_ids)).any():
        raise ValueError("Checkpoint contains a neighbor index outside the sensor list")
    if not np.array_equal(neighbors[:, 0], np.arange(len(sensor_ids))):
        raise ValueError("The target sensor must be first in each checkpoint window")
    if (neighbors[:, 1:] == neighbors[:, :1]).any():
        raise ValueError("The target sensor must not reappear among its neighbors")

    return model, means, stds, sensor_ids, neighbors


def load_validation():
    """Recover original observations at the validation split keys."""
    keys = pd.read_csv(PROCESSED_DIR / "val.csv", usecols=KEY_COLUMNS)
    observations = pd.read_csv(PROCESSED_DIR / "sensor_data.csv", usecols=INPUT_COLUMNS)
    validation = keys.merge(
        observations, on=KEY_COLUMNS, how="left", validate="one_to_one", indicator=True
    )
    if (validation["_merge"] != "both").any():
        raise ValueError("Some validation rows are missing from sensor_data.csv")

    return validation[INPUT_COLUMNS]


def quality_rules(observations):
    """Return separate physical-invalidity and heuristic reasons for each cell."""
    values = observations[MEASUREMENT_COLUMNS]
    invalid = pd.DataFrame("", index=values.index, columns=MEASUREMENT_COLUMNS)
    heuristic = invalid.copy()

    for feature in MEASUREMENT_COLUMNS:
        invalid.loc[np.isinf(values[feature]), feature] = "nonfinite_reading"

    humidity = values["humidity"]
    light = values["light"]
    voltage = values["voltage"]
    temperature = values["temperature"]

    invalid.loc[np.isfinite(humidity) & ~humidity.between(0, 100), "humidity"] = (
        "humidity_outside_physical_range"
    )
    invalid.loc[np.isfinite(light) & (light < 0), "light"] = "negative_light"

    heuristic.loc[
        np.isfinite(temperature) & ~temperature.between(-20, 60), "temperature"
    ] = "temperature_outside_project_range"
    heuristic.loc[
        np.isfinite(voltage) & ~voltage.between(2, 3), "voltage"
    ] = "voltage_outside_deployment_range"

    return invalid, heuristic


def prepare_inputs(observations, invalid, means, stds, sensor_ids):
    """Remove physically invalid context and align it to the checkpoint sensor order."""
    if observations.empty:
        raise ValueError("No sensor observations")
    if observations.duplicated(KEY_COLUMNS).any():
        raise ValueError("Input contains duplicate time and sensor pairs")
    if not observations["nodeid"].isin(sensor_ids).all():
        raise ValueError("Input contains a sensor absent from the checkpoint")
    if not (observations["time"] % TIME_STEP == 0).all():
        raise ValueError("Input timestamps must align to the 30-second grid")

    context = observations.copy()
    context[MEASUREMENT_COLUMNS] = observations[MEASUREMENT_COLUMNS].mask(invalid.ne(""))
    values, availability, times, _ = build_sensor_arrays(context, sensor_ids)
    if len(times) < WINDOW_LENGTH:
        raise ValueError(f"GRU context needs at least {WINDOW_LENGTH} timesteps")

    return normalize_measurements(values, means, stds), availability, times


def predict_targets(model, values, availability, times, sensor_ids, neighbors,
                    means, stds, targets, batch_size=BATCH_SIZE):
    """Hide each requested reading in an independent window before predicting it."""
    time_indices = pd.Index(times).get_indexer(targets["time"])
    sensor_indices = pd.Index(sensor_ids).get_indexer(targets["nodeid"])
    feature_indices = pd.Index(MEASUREMENT_COLUMNS).get_indexer(targets["feature"])
    if (time_indices < 0).any() or (sensor_indices < 0).any() or (feature_indices < 0).any():
        raise ValueError("Unknown target time, sensor, or feature")
    if not availability[time_indices, sensor_indices, feature_indices].all():
        raise ValueError("Residual targets must be observed and valid")

    predictions = np.empty(len(targets), dtype=float)
    with torch.inference_mode():
        for start in range(0, len(targets), batch_size):
            stop = start + batch_size
            target_times = time_indices[start:stop]
            target_features = feature_indices[start:stop]
            window_starts = np.clip(
                target_times - WINDOW_LENGTH // 2, 0, len(times) - WINDOW_LENGTH
            )
            target_steps = target_times - window_starts
            window_times = window_starts[:, None] + np.arange(WINDOW_LENGTH)
            window_sensors = neighbors[sensor_indices[start:stop]]

            # Advanced indexing copies each window, so hiding a target cannot alter context.
            measurements = values[window_times[:, :, None], window_sensors[:, None, :]]
            flags = availability[window_times[:, :, None], window_sensors[:, None, :]]
            batch_rows = np.arange(len(target_times))

            measurements[batch_rows, target_steps, 0, target_features] = 0.0
            flags[batch_rows, target_steps, 0, target_features] = False

            inputs = np.concatenate([measurements, flags.astype(np.float32)], axis=3)
            inputs = inputs.reshape(len(target_times), WINDOW_LENGTH, -1)
            outputs = model(torch.from_numpy(inputs)).numpy()
            normalized = outputs[batch_rows, target_steps, target_features]
            predictions[start:stop] = normalized * stds[target_features] + means[target_features]

    if not np.isfinite(predictions).all():
        raise ValueError("GRU produced a nonfinite prediction")

    return predictions


def reading_rows(observations, invalid, heuristic):
    """Expand sensor rows to individual readings with their rule reasons."""
    readings = observations.melt(
        id_vars=KEY_COLUMNS, value_vars=MEASUREMENT_COLUMNS,
        var_name="feature", value_name="observed_value"
    )
    readings["invalid_reason"] = invalid[MEASUREMENT_COLUMNS].to_numpy().T.ravel()
    readings["heuristic_reason"] = heuristic[MEASUREMENT_COLUMNS].to_numpy().T.ravel()

    return readings


def calibrate(model, means, stds, sensor_ids, neighbors):
    """Estimate feature thresholds from a seeded sample of validation residuals."""
    observations = load_validation()
    invalid, heuristic = quality_rules(observations)
    values, availability, times = prepare_inputs(
        observations, invalid, means, stds, sensor_ids
    )
    readings = reading_rows(observations, invalid, heuristic)
    rng = np.random.default_rng(CALIBRATION_SEED)

    print(f"checkpoint sha256: {hashlib.sha256(CHECKPOINT_PATH.read_bytes()).hexdigest()}")
    print(f"validation timestamps: {times[0]} to {times[-1]}")
    print(f"calibration quantile: {CALIBRATION_QUANTILE}")

    thresholds = {}
    for feature in MEASUREMENT_COLUMNS:
        eligible = readings[
            readings["feature"].eq(feature) & readings["observed_value"].notna()
            & readings["invalid_reason"].eq("") & readings["heuristic_reason"].eq("")
        ]
        if eligible.empty:
            raise ValueError(f"No eligible validation readings for {feature}")
        positions = rng.choice(
            len(eligible), size=min(CALIBRATION_SAMPLE_SIZE, len(eligible)), replace=False
        )
        targets = eligible.iloc[positions]
        predictions = predict_targets(
            model, values, availability, times, sensor_ids, neighbors, means, stds, targets
        )
        residuals = targets["observed_value"].to_numpy() - predictions
        thresholds[feature] = float(np.quantile(np.abs(residuals), CALIBRATION_QUANTILE))
        print(f"{feature}: {len(targets):,} calibration readings")

    print(f"RESIDUAL_THRESHOLDS = {thresholds!r}")


def score_readings(readings, model, values, availability, times, sensor_ids,
                   neighbors, means, stds):
    """Score observed, physically valid readings and combine all alert layers."""
    readings = readings.copy()
    missing = readings["observed_value"].isna()
    invalid = readings["invalid_reason"].ne("")
    eligible = ~missing & ~invalid

    readings["score_status"] = "scored"
    readings.loc[missing, "score_status"] = "missing"
    readings.loc[invalid, "score_status"] = "invalid"
    readings["predicted_value"] = np.nan
    readings.loc[eligible, "predicted_value"] = predict_targets(
        model, values, availability, times, sensor_ids, neighbors, means, stds,
        readings.loc[eligible]
    )
    readings["residual"] = readings["observed_value"] - readings["predicted_value"]
    readings["anomaly_score"] = readings["residual"].abs()
    readings["threshold"] = readings["feature"].map(RESIDUAL_THRESHOLDS).where(eligible)
    readings["residual_flag"] = readings["anomaly_score"] > readings["threshold"]
    readings["is_anomaly"] = invalid | readings["heuristic_reason"].ne("") | readings["residual_flag"]

    return readings


def detect(model, means, stds, sensor_ids, neighbors, start_time, end_time, output_path, input_path):
    """Score a time range using full surrounding context and publish the CSV atomically."""
    if set(RESIDUAL_THRESHOLDS) != set(MEASUREMENT_COLUMNS):
        raise ValueError("Residual thresholds must cover every measurement feature")
    for feature, threshold in RESIDUAL_THRESHOLDS.items():
        if not np.isfinite(threshold) or threshold < 0:
            raise ValueError(f"Invalid residual threshold for {feature}")

    observations = pd.read_csv(input_path, usecols=INPUT_COLUMNS)
    observations = observations.sort_values(KEY_COLUMNS)
    selected = pd.Series(True, index=observations.index)
    if start_time is not None:
        selected &= observations["time"] >= start_time
    if end_time is not None:
        selected &= observations["time"] < end_time
    scoring_times = observations.loc[selected, "time"].unique()
    if len(scoring_times) == 0:
        raise ValueError("No observations in the requested scoring range")

    invalid, heuristic = quality_rules(observations)
    values, availability, times = prepare_inputs(
        observations, invalid, means, stds, sensor_ids
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_rows = 0
    total_flags = 0

    temporary_path = None
    try:
        with NamedTemporaryFile(mode="w", dir=output_path.parent, delete=False) as output:
            temporary_path = Path(output.name)
            for start in range(0, len(scoring_times), TIMESTAMPS_PER_CHUNK):
                chunk_times = scoring_times[start:start + TIMESTAMPS_PER_CHUNK]
                in_chunk = observations["time"].between(chunk_times[0], chunk_times[-1])
                readings = reading_rows(
                    observations.loc[in_chunk], invalid.loc[in_chunk], heuristic.loc[in_chunk]
                )
                results = score_readings(
                    readings, model, values, availability, times, sensor_ids,
                    neighbors, means, stds
                ).sort_values(KEY_COLUMNS + ["feature"])
                results.to_csv(output, index=False, header=(start == 0))
                total_rows += len(results)
                total_flags += int(results["is_anomaly"].sum())
        temporary_path.replace(output_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    print(f"saved {total_rows:,} readings with {total_flags:,} flags to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Detect sensor anomalies with a saved GRU")
    parser.add_argument("--calibrate", action=argparse.BooleanOptionalAction,
                        default=CALIBRATE_ONLY, help="print validation thresholds")
    parser.add_argument("--start-time", type=int, default=START_TIME,
                        help="first elapsed second to score, inclusive")
    parser.add_argument("--end-time", type=int, default=END_TIME,
                        help="elapsed second at which to stop scoring, exclusive")
    parser.add_argument("--input", type=Path, default=INPUT_PATH, help="processed observations CSV path")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH, help="detection CSV path")
    args = parser.parse_args()
    if args.start_time is not None and args.end_time is not None:
        if args.start_time >= args.end_time:
            parser.error("--start-time must be earlier than --end-time")

    torch.set_num_threads(2)
    model, means, stds, sensor_ids, neighbors = load_checkpoint()
    if args.calibrate:
        calibrate(model, means, stds, sensor_ids, neighbors)
    else:
        detect(model, means, stds, sensor_ids, neighbors,
               args.start_time, args.end_time, args.output, args.input)


if __name__ == "__main__":
    main()
