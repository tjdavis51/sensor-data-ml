# anomaly detection with physical rules and gru reconstruction

# input from the processed 30-second sensor data and the saved gru checkpoint
# physical and heuristic checks for each temperature humidity light and voltage reading
# leave-one-out prediction using the surrounding readings and the four nearest sensors
# feature thresholds from the 99th percentile of validation reconstruction errors
# output with one row per time sensor and feature and separate reasons for each flag
# run settings at the top of this file and optional overrides from the command line

# load the helpers for arguments paths and temporary output
import argparse
import hashlib
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

# load the libraries for arrays csv files and gru inference
import numpy as np
import pandas as pd
import torch

# find the repo root from the location of this file
ROOT = Path(__file__).resolve().parents[1]
# add the import path only for direct script runs
if __package__ in (None, ""):
    # let python find src when the script is run by file path
    sys.path.insert(0, str(ROOT))

# reuse the reconstruction model and its input conventions
from src.models.gru.gru import (
    MEASUREMENT_COLUMNS,
    SensorGRU,
    TIME_STEP,
    build_sensor_arrays,
    normalize_measurements,
)


# default run settings for python anomaly_detection/anomaly_detection.py
PROCESSED_DIR = ROOT / "data/processed"
# use the later unlabeled period for the default detector run
INPUT_PATH = PROCESSED_DIR / "anomaly_period.csv"
# use the saved reconstruction weights and training metadata
CHECKPOINT_PATH = ROOT / "src/models/gru/checkpoints/best_model.pt"
# write detections to the local results folder
OUTPUT_PATH = ROOT / "anomaly_detection/results/detections.csv"

# start inclusive and end exclusive in elapsed seconds
# set both to none to score the entire input
START_TIME = 1728000
# score ten minutes by default for a quick local run
END_TIME = 1728600
# score data by default and switch modes to reproduce the thresholds
CALIBRATE_ONLY = False

# use the same window length as gru reconstruction
WINDOW_LENGTH = 40
# group independent predictions into batches for inference
BATCH_SIZE = 128
# limit the number of timestamps converted to output rows at once
TIMESTAMPS_PER_CHUNK = 120

# use the 99th percentile of absolute leave-one-out residuals in native measurement units
# sample of 4096 readings per feature from validation timestamps 1187880 through 1423530
# checkpoint sha256 ebee071a63bdbf51bc4ced837fcd80822b14fb026a3292e04c9d49a2d61c7644
# reproduce the values with --calibrate
CALIBRATION_QUANTILE = 0.99
# cap the number of calibration readings per feature
CALIBRATION_SAMPLE_SIZE = 4096
# use a fixed seed to reproduce the calibration sample
CALIBRATION_SEED = 42
# store each threshold in the original units of its feature
RESIDUAL_THRESHOLDS = {
    "temperature": 1.500306009521519,
    "humidity": 2.6222635302734463,
    "light": 284.90152954101563,
    "voltage": 0.14568523091125507,
}

# identify each source row with its elapsed time and sensor id
KEY_COLUMNS = ["time", "nodeid"]
# read the source keys and measurement columns used by the detector
INPUT_COLUMNS = KEY_COLUMNS + MEASUREMENT_COLUMNS


# load the model and the training information needed for inference
def load_checkpoint():
    # load the checkpoint tensors on the cpu
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    # create the same gru architecture used in reconstruction
    model = SensorGRU()
    # fill the model with the saved weights
    model.load_state_dict(checkpoint["model_state_dict"])
    # put the model in inference mode
    model.eval()

    # recover the training mean for each measurement
    means = checkpoint["means"].numpy()
    # recover the training standard deviation for each measurement
    stds = checkpoint["stds"].numpy()
    # recover the sensor order used when training
    sensor_ids = checkpoint["sensor_ids"].numpy()
    # recover the target and neighbor indices in that order
    neighbors = checkpoint["neighbors"].numpy()

    # check the normalization and sensor order used to hide each target
    if not np.isfinite(means).all() or not np.isfinite(stds).all() or (stds <= 0).any():
        raise ValueError("Checkpoint normalization must have finite means and positive standard deviations")
    # require one target sensor and four neighbors per window
    if neighbors.shape != (len(sensor_ids), 5):
        raise ValueError("Checkpoint must specify the target sensor and four neighbors")
    # reject neighbor indices outside the checkpoint sensor list
    if (neighbors < 0).any() or (neighbors >= len(sensor_ids)).any():
        raise ValueError("Checkpoint contains a neighbor index outside the sensor list")
    # verify the position used to hide the target reading
    if not np.array_equal(neighbors[:, 0], np.arange(len(sensor_ids))):
        raise ValueError("The target sensor must be first in each checkpoint window")
    # prevent the target from remaining visible in a neighbor slot
    if (neighbors[:, 1:] == neighbors[:, :1]).any():
        raise ValueError("The target sensor must not reappear among its neighbors")

    # return the model and the settings needed to prepare windows
    return model, means, stds, sensor_ids, neighbors


# find the original measurements for the validation keys
def load_validation():
    # recover measurements blanked by the reconstruction split
    keys = pd.read_csv(PROCESSED_DIR / "val.csv", usecols=KEY_COLUMNS)
    # read the aggregates before suspicious values were blanked by splitting
    observations = pd.read_csv(PROCESSED_DIR / "sensor_data.csv", usecols=INPUT_COLUMNS)
    # match the original measurements to the time and sensor of each validation row
    validation = keys.merge(
        observations, on=KEY_COLUMNS, how="left", validate="one_to_one", indicator=True
    )
    # fail if a validation key has no original row
    if (validation["_merge"] != "both").any():
        raise ValueError("Some validation rows are missing from sensor_data.csv")

    # keep only the keys and measurements for scoring
    return validation[INPUT_COLUMNS]


# apply the physical and heuristic checks to each measurement
def quality_rules(observations):
    # use separate reason columns so a heuristic and a residual can both be flagged
    values = observations[MEASUREMENT_COLUMNS]
    # start with no physical reason on any cell
    invalid = pd.DataFrame("", index=values.index, columns=MEASUREMENT_COLUMNS)
    # give heuristic reasons a separate table with the same layout
    heuristic = invalid.copy()

    # check all four features for infinite values
    for feature in MEASUREMENT_COLUMNS:
        invalid.loc[np.isinf(values[feature]), feature] = "nonfinite_reading"

    # pull out the individual measurement columns for the range checks
    humidity = values["humidity"]
    light = values["light"]
    voltage = values["voltage"]
    temperature = values["temperature"]

    # flag relative humidity outside the physical range
    invalid.loc[np.isfinite(humidity) & ~humidity.between(0, 100), "humidity"] = (
        "humidity_outside_physical_range"
    )
    # flag negative illuminance
    invalid.loc[np.isfinite(light) & (light < 0), "light"] = "negative_light"

    # use the project's temperature bounds and the documented deployment voltage range
    heuristic.loc[
        np.isfinite(temperature) & ~temperature.between(-20, 60), "temperature"
    ] = "temperature_outside_project_range"
    # flag voltage outside the documented deployment range
    heuristic.loc[
        np.isfinite(voltage) & ~voltage.between(2, 3), "voltage"
    ] = "voltage_outside_deployment_range"

    # return both sets of reasons without altering the observations
    return invalid, heuristic


# build normalized gru context from valid measurements
def prepare_inputs(observations, invalid, means, stds, sensor_ids):
    # require observations to build the time grid
    if observations.empty:
        raise ValueError("No sensor observations")
    # require one source row per time and sensor before building arrays
    if observations.duplicated(KEY_COLUMNS).any():
        raise ValueError("Input contains duplicate time and sensor pairs")
    # check that every input sensor exists in the checkpoint
    if not observations["nodeid"].isin(sensor_ids).all():
        raise ValueError("Input contains a sensor absent from the checkpoint")
    # check that input times match the interval used in training
    if not (observations["time"] % TIME_STEP == 0).all():
        raise ValueError("Input timestamps must align to the 30-second grid")

    # keep invalid readings in the source table but remove them from every model window
    context = observations.copy()
    # replace physically invalid cells with missing values in the model input
    context[MEASUREMENT_COLUMNS] = observations[MEASUREMENT_COLUMNS].mask(invalid.ne(""))
    # align the context to the checkpoint sensor order and the time grid
    values, availability, times, _ = build_sensor_arrays(context, sensor_ids)
    # require enough context to make a complete window
    if len(times) < WINDOW_LENGTH:
        raise ValueError(f"GRU context needs at least {WINDOW_LENGTH} timesteps")

    # normalize with the saved training values and retain the availability flags
    return normalize_measurements(values, means, stds), availability, times


# predict each observed target after hiding it from its own window
def predict_targets(model, values, availability, times, sensor_ids, neighbors,
                    means, stds, targets, batch_size=BATCH_SIZE):
    # find the array index for each target timestamp
    time_indices = pd.Index(times).get_indexer(targets["time"])
    # find the array index for each target sensor
    sensor_indices = pd.Index(sensor_ids).get_indexer(targets["nodeid"])
    # find the array index for each target feature
    feature_indices = pd.Index(MEASUREMENT_COLUMNS).get_indexer(targets["feature"])
    # reject unknown keys before using them as array indices
    if (time_indices < 0).any() or (sensor_indices < 0).any() or (feature_indices < 0).any():
        raise ValueError("Unknown target time, sensor, or feature")
    # require each target to be an observed reading eligible for residual scoring
    if not availability[time_indices, sensor_indices, feature_indices].all():
        raise ValueError("Residual targets must be observed and valid")

    # reserve one result in the original target order
    predictions = np.empty(len(targets), dtype=float)
    # turn off gradient tracking for all inference batches
    with torch.inference_mode():
        # process the target requests one batch at a time
        for start in range(0, len(targets), batch_size):
            # find the upper bound of the batch
            stop = start + batch_size
            # take the time indices for the current targets
            target_times = time_indices[start:stop]
            # take the feature indices for the current targets
            target_features = feature_indices[start:stop]
            # center the windows and shift them inward at the data edges
            window_starts = np.clip(
                target_times - WINDOW_LENGTH // 2, 0, len(times) - WINDOW_LENGTH
            )
            # find the target timestep within each window
            target_steps = target_times - window_starts
            # expand each start into a full list of timesteps
            window_times = window_starts[:, None] + np.arange(WINDOW_LENGTH)
            # look up the target sensor and its four neighbors
            window_sensors = neighbors[sensor_indices[start:stop]]

            # advanced indexing creates independent windows for each target
            measurements = values[window_times[:, :, None], window_sensors[:, None, :]]
            # copy the matching availability flags for each window
            flags = availability[window_times[:, :, None], window_sensors[:, None, :]]
            # use one batch index for each independently hidden target
            batch_rows = np.arange(len(target_times))

            # the first sensor in each window is the target sensor
            measurements[batch_rows, target_steps, 0, target_features] = 0.0
            # mark the hidden target as unavailable to the model
            flags[batch_rows, target_steps, 0, target_features] = False

            # place the values and availability flags side by side for each sensor
            inputs = np.concatenate([measurements, flags.astype(np.float32)], axis=3)
            # flatten the sensor inputs at each timestep into the gru input layout
            inputs = inputs.reshape(len(target_times), WINDOW_LENGTH, -1)
            # predict four measurements for each timestep
            outputs = model(torch.from_numpy(inputs)).numpy()
            # select the predicted feature at the target timestep
            normalized = outputs[batch_rows, target_steps, target_features]
            # convert the selected predictions back to their original units
            predictions[start:stop] = normalized * stds[target_features] + means[target_features]

    # reject incomplete or nonfinite model output
    if not np.isfinite(predictions).all():
        raise ValueError("GRU produced a nonfinite prediction")

    # return the predictions in the same order as the targets
    return predictions


# turn each sensor record into one row per feature
def reading_rows(observations, invalid, heuristic):
    # keep the same row and feature order in all three tables
    readings = observations.melt(
        id_vars=KEY_COLUMNS, value_vars=MEASUREMENT_COLUMNS,
        var_name="feature", value_name="observed_value"
    )
    # attach the physical reasons in the same order as the feature rows
    readings["invalid_reason"] = invalid[MEASUREMENT_COLUMNS].to_numpy().T.ravel()
    # attach the heuristic reasons in that order too
    readings["heuristic_reason"] = heuristic[MEASUREMENT_COLUMNS].to_numpy().T.ravel()

    # return the readings and their rule reasons
    return readings


# reproduce the validation-based residual thresholds
def calibrate(model, means, stds, sensor_ids, neighbors):
    # recover the original readings for the validation interval
    observations = load_validation()
    # apply the same rules used during detection
    invalid, heuristic = quality_rules(observations)
    # prepare the gru context with physically invalid values removed
    values, availability, times = prepare_inputs(
        observations, invalid, means, stds, sensor_ids
    )
    # make a row for each candidate reading
    readings = reading_rows(observations, invalid, heuristic)
    # start the seeded sampler for choosing validation readings
    rng = np.random.default_rng(CALIBRATION_SEED)

    # report the exact weights used for these thresholds
    print(f"checkpoint sha256: {hashlib.sha256(CHECKPOINT_PATH.read_bytes()).hexdigest()}")
    # report the validation time range
    print(f"validation timestamps: {times[0]} to {times[-1]}")
    # report the percentile used for each feature
    print(f"calibration quantile: {CALIBRATION_QUANTILE}")

    # collect one threshold for each feature
    thresholds = {}
    # calibrate each feature in its own measurement units
    for feature in MEASUREMENT_COLUMNS:
        # keep observed readings that pass both rule layers as calibration targets
        eligible = readings[
            readings["feature"].eq(feature) & readings["observed_value"].notna()
            & readings["invalid_reason"].eq("") & readings["heuristic_reason"].eq("")
        ]
        # fail if there are no eligible readings for this feature
        if eligible.empty:
            raise ValueError(f"No eligible validation readings for {feature}")
        # choose up to the configured sample size without replacement
        positions = rng.choice(
            len(eligible), size=min(CALIBRATION_SAMPLE_SIZE, len(eligible)), replace=False
        )
        # take the validation targets selected by the sampler
        targets = eligible.iloc[positions]
        # reconstruct those readings without exposing the target values
        predictions = predict_targets(
            model, values, availability, times, sensor_ids, neighbors, means, stds, targets
        )
        # subtract each prediction from its observed value
        residuals = targets["observed_value"].to_numpy() - predictions
        # set the threshold from the percentile of the absolute errors
        thresholds[feature] = float(np.quantile(np.abs(residuals), CALIBRATION_QUANTILE))
        # report how many values contributed to this threshold
        print(f"{feature}: {len(targets):,} calibration readings")

    # print the constants in a form that can be copied into this file
    print(f"RESIDUAL_THRESHOLDS = {thresholds!r}")


# attach residual scores and status information to the reading rows
def score_readings(readings, model, values, availability, times, sensor_ids,
                   neighbors, means, stds):
    # copy the reading rows before adding result columns
    readings = readings.copy()
    # identify measurements with no observed value
    missing = readings["observed_value"].isna()
    # identify readings excluded by a physical rule
    invalid = readings["invalid_reason"].ne("")
    # score only observed readings that pass the physical checks
    eligible = ~missing & ~invalid

    # start with the status used for eligible readings
    readings["score_status"] = "scored"
    # give empty readings their own status
    readings.loc[missing, "score_status"] = "missing"
    # give physically invalid readings a separate status
    readings.loc[invalid, "score_status"] = "invalid"
    # leave predictions blank until they are filled for eligible readings
    readings["predicted_value"] = np.nan
    # reconstruct only the readings eligible for residual scoring
    readings.loc[eligible, "predicted_value"] = predict_targets(
        model, values, availability, times, sensor_ids, neighbors, means, stds,
        readings.loc[eligible]
    )
    # compute the signed difference in the original measurement units
    readings["residual"] = readings["observed_value"] - readings["predicted_value"]
    # use the absolute difference as the anomaly score
    readings["anomaly_score"] = readings["residual"].abs()
    # look up the feature threshold only for scored readings
    readings["threshold"] = readings["feature"].map(RESIDUAL_THRESHOLDS).where(eligible)
    # flag scores strictly above their own feature threshold
    readings["residual_flag"] = readings["anomaly_score"] > readings["threshold"]
    # combine either rule signal with the residual flag
    readings["is_anomaly"] = invalid | readings["heuristic_reason"].ne("") | readings["residual_flag"]

    # return the readings with reasons scores and flags
    return readings


# score the requested range and save the completed detections table
def detect(model, means, stds, sensor_ids, neighbors, start_time, end_time, output_path, input_path):
    # require a usable threshold for every feature before scoring
    if set(RESIDUAL_THRESHOLDS) != set(MEASUREMENT_COLUMNS):
        raise ValueError("Residual thresholds must cover every measurement feature")
    # check each configured feature threshold
    for feature, threshold in RESIDUAL_THRESHOLDS.items():
        # reject thresholds that cannot be used for a residual comparison
        if not np.isfinite(threshold) or threshold < 0:
            raise ValueError(f"Invalid residual threshold for {feature}")

    # load the processed measurements and their source keys
    observations = pd.read_csv(input_path, usecols=INPUT_COLUMNS)
    # sort by time and sensor for predictable output order
    observations = observations.sort_values(KEY_COLUMNS)
    # start with all observations in the requested range
    selected = pd.Series(True, index=observations.index)
    # apply the inclusive lower bound when configured
    if start_time is not None:
        selected &= observations["time"] >= start_time
    # apply the exclusive upper bound when configured
    if end_time is not None:
        selected &= observations["time"] < end_time
    # collect the distinct timestamps to score
    scoring_times = observations.loc[selected, "time"].unique()
    # stop before inference if the requested range has no observations
    if len(scoring_times) == 0:
        raise ValueError("No observations in the requested scoring range")

    # retain context on both sides of the output range
    invalid, heuristic = quality_rules(observations)
    # build the context arrays from the full input around the scoring range
    values, availability, times = prepare_inputs(
        observations, invalid, means, stds, sensor_ids
    )
    # create the results folder if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # start the output reading count
    total_rows = 0
    # start the combined alert count
    total_flags = 0

    # write in chunks and publish the csv after all predictions succeed
    temporary_path = None
    # remove the temporary output even if a chunk fails
    try:
        # open the temporary csv in the same folder as the final output
        with NamedTemporaryFile(mode="w", dir=output_path.parent, delete=False) as output:
            # keep its path for the final rename or cleanup
            temporary_path = Path(output.name)
            # take a fixed number of scoring timestamps per chunk
            for start in range(0, len(scoring_times), TIMESTAMPS_PER_CHUNK):
                # select the timestamps for this chunk
                chunk_times = scoring_times[start:start + TIMESTAMPS_PER_CHUNK]
                # find the sensor rows that belong to those timestamps
                in_chunk = observations["time"].between(chunk_times[0], chunk_times[-1])
                # expand the source rows and rule reasons into individual readings
                readings = reading_rows(
                    observations.loc[in_chunk], invalid.loc[in_chunk], heuristic.loc[in_chunk]
                )
                # score the readings and sort them before writing
                results = score_readings(
                    readings, model, values, availability, times, sensor_ids,
                    neighbors, means, stds
                ).sort_values(KEY_COLUMNS + ["feature"])
                # write the csv header only for the first chunk
                results.to_csv(output, index=False, header=(start == 0))
                # add the number of emitted readings
                total_rows += len(results)
                # add the number flagged by at least one layer
                total_flags += int(results["is_anomaly"].sum())
        # publish the complete csv at the requested output path
        temporary_path.replace(output_path)
    finally:
        # clean up only if a temporary path was created
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    # report the output location and flag totals
    print(f"saved {total_rows:,} readings with {total_flags:,} flags to {output_path}")


# read the run settings and choose between calibration and detection
def main():
    # set up the optional overrides for the constants above
    parser = argparse.ArgumentParser(description="Detect sensor anomalies with a saved GRU")
    # allow the calibration mode to be turned on or off
    parser.add_argument("--calibrate", action=argparse.BooleanOptionalAction,
                        default=CALIBRATE_ONLY, help="print validation thresholds")
    # allow a different first elapsed second
    parser.add_argument("--start-time", type=int, default=START_TIME,
                        help="first elapsed second to score, inclusive")
    # allow a different stopping time
    parser.add_argument("--end-time", type=int, default=END_TIME,
                        help="elapsed second at which to stop scoring, exclusive")
    # allow a different processed input csv
    parser.add_argument("--input", type=Path, default=INPUT_PATH, help="processed observations CSV path")
    # allow a different results path
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH, help="detection CSV path")
    # read the command line and fill in any defaults
    args = parser.parse_args()
    # validate the ordering when both time bounds are set
    if args.start_time is not None and args.end_time is not None:
        if args.start_time >= args.end_time:
            parser.error("--start-time must be earlier than --end-time")

    # use a small thread pool for batched cpu inference
    torch.set_num_threads(2)
    # load the model and its input settings once
    model, means, stds, sensor_ids, neighbors = load_checkpoint()
    # use the same checkpoint for either mode
    if args.calibrate:
        # print the thresholds from the validation sample
        calibrate(model, means, stds, sensor_ids, neighbors)
    else:
        # score the selected observations with the frozen thresholds
        detect(model, means, stds, sensor_ids, neighbors,
               args.start_time, args.end_time, args.output, args.input)


# run the entry point when this file is started directly
if __name__ == "__main__":
    main()
