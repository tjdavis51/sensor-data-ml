# reconstruct hidden sensor readings with interpolation and a saved gru
#
# input from a shared masked split and its target keys
# interpolation between visible readings from the same sensor and feature
# saved gru prediction when either side lacks a nearby reading
# output with one prediction per target for the shared benchmark
# run settings at the top of this file for a direct python run

from pathlib import Path

import numpy as np
import pandas as pd
import torch

# support running the file directly or importing it from the repo
if __package__:
    from . import gru
else:
    import gru


# select the masked split and the fallback checkpoint filename
DATASET = "test"
GRU_CHECKPOINT_NAME = "best_model.pt"
MAX_GAP_STEPS = 20

# locate the data and keep these predictions separate from the original gru results
ROOT = Path(__file__).resolve().parents[3]
MASKED_DIR = ROOT / "data/masked"
CHECKPOINT_DIR = ROOT / "src/models/gru/checkpoints"
PREDICTIONS_PATH = ROOT / "src/models/gru/results/interpolation/predictions.csv"
TARGET_COLUMNS = ["target_id", "time", "nodeid", "feature"]


def interpolate_targets(values, availability, time_indices, sensor_indices, feature_indices):
    # start with blank predictions for targets that may need the gru
    predictions = np.full(len(time_indices), np.nan)
    steps = np.arange(len(values))[:, None]

    # look for nearby visible readings separately for each feature
    for feature in range(len(gru.MEASUREMENT_COLUMNS)):
        rows = np.flatnonzero(feature_indices == feature)
        target_times = time_indices[rows]
        target_sensors = sensor_indices[rows]
        visible = availability[:, :, feature]

        # carry the nearest visible index from each direction along the time grid
        previous = np.maximum.accumulate(np.where(visible, steps, -len(values)), axis=0)
        following = np.minimum.accumulate(
            np.where(visible, steps, 2 * len(values))[::-1], axis=0
        )[::-1]

        # find the before and after indices for each requested time and sensor
        left = previous[target_times, target_sensors]
        right = following[target_times, target_sensors]
        left_gap = target_times - left
        right_gap = right - target_times

        # require a visible reading within the configured distance on each side
        supported = ((left_gap > 0) & (left_gap <= MAX_GAP_STEPS)
                     & (right_gap > 0) & (right_gap <= MAX_GAP_STEPS))

        # look up only valid endpoints so missing indices never reach the value array
        before = values[left[supported], target_sensors[supported], feature]
        after = values[right[supported], target_sensors[supported], feature]
        left_distance = left_gap[supported]
        right_distance = right_gap[supported]

        # give the closer reading more weight using its opposite endpoint distance
        predictions[rows[supported]] = (
            (before * right_distance + after * left_distance)
            / (left_distance + right_distance)
        )

    return predictions


def main():
    # load the masked measurements and target identities without the hidden values
    frame = pd.read_csv(MASKED_DIR / f"{DATASET}_masked.csv")
    targets = pd.read_csv(MASKED_DIR / f"{DATASET}_answers.csv", usecols=TARGET_COLUMNS)
    if targets.empty or not targets["target_id"].is_unique:
        raise ValueError("Targets must be nonempty with unique IDs")

    # read the sensor order used by the selected checkpoint
    checkpoint_path = CHECKPOINT_DIR / GRU_CHECKPOINT_NAME
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    sensor_ids = checkpoint["sensor_ids"].numpy()

    # align the visible readings to the time grid and the checkpoint sensor order
    values, availability, times, _ = gru.build_sensor_arrays(frame, sensor_ids)
    time_indices = pd.Index(times).get_indexer(targets["time"])
    sensor_indices = pd.Index(sensor_ids).get_indexer(targets["nodeid"])
    feature_indices = pd.Index(gru.MEASUREMENT_COLUMNS).get_indexer(targets["feature"])

    # reject unknown targets and confirm that target cells are hidden
    if min(time_indices.min(), sensor_indices.min(), feature_indices.min()) < 0:
        raise ValueError(f"Unknown target time sensor or feature in {DATASET}")
    if availability[time_indices, sensor_indices, feature_indices].any():
        raise ValueError(f"Target values are visible in the {DATASET} mask")

    # use interpolation wherever both nearby readings are available
    predictions = interpolate_targets(values, availability, time_indices, sensor_indices, feature_indices)
    needs_gru = np.isnan(predictions)

    if needs_gru.any():
        # reuse the original gru inference path only for the remaining targets
        torch.set_num_threads(2)
        model = gru.SensorGRU()
        model.load_state_dict(checkpoint["model_state_dict"])
        means = checkpoint["means"].numpy()
        stds = checkpoint["stds"].numpy()
        neighbors = checkpoint["neighbors"].numpy()
        normalized = gru.normalize_measurements(values, means, stds)
        fallback_targets = targets.loc[needs_gru]
        fallback = gru.predict_targets(
            model, normalized, availability, neighbors, times,
            sensor_ids, fallback_targets, means, stds
        )

        # match the fallback predictions by target id
        predictions[needs_gru] = fallback_targets["target_id"].map(
            fallback.set_index("target_id")["prediction"]
        ).to_numpy()

    # keep one finite prediction for each shared target
    if not np.isfinite(predictions).all():
        raise ValueError("Some targets did not receive finite predictions")
    output = pd.DataFrame({"target_id": targets["target_id"], "prediction": predictions})

    # write the benchmark format and report how often each method was used
    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(PREDICTIONS_PATH, index=False)
    print(f"Interpolation: {(~needs_gru).sum():,} targets")
    print(f"GRU fallback ({GRU_CHECKPOINT_NAME}): {needs_gru.sum():,} targets")
    print(f"Saved {len(output):,} predictions to {PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()
