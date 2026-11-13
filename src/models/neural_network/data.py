"""Masked windows using the GRU's existing observation and spatial contracts."""

import numpy as np
import pandas as pd

from src.models.gru.gru import (
    build_sensor_arrays,
    normalize_measurements,
)
from src.utils.masking import MEASUREMENT_COLUMNS

WINDOW_LENGTH = 40
INPUT_SIZE = WINDOW_LENGTH * 5 * 8 + WINDOW_LENGTH


class MaskedWindows:
    """Keep target identities separate from labels and construct batches on demand."""

    def __init__(self, frame, targets, sensor_ids, neighbors, means, stds):
        if frame.duplicated(["time", "nodeid"]).any():
            raise ValueError("Duplicate time/node pairs")
        if not frame.nodeid.isin(sensor_ids).all():
            raise ValueError("Unknown sensor in input data")
        values, self.available, times, _ = build_sensor_arrays(frame, sensor_ids)
        if len(times) < WINDOW_LENGTH:
            raise ValueError("Split must contain at least 40 time steps")
        if np.isinf(values).any():
            raise ValueError("Infinite measurement in inputs")
        self.values = normalize_measurements(values, means, stds)
        self.neighbors = neighbors
        self.target_ids = targets.target_id.to_numpy()
        if targets.empty or targets.target_id.duplicated().any():
            raise ValueError("Targets must be nonempty with unique IDs")
        sensor_indices = pd.Index(sensor_ids).get_indexer(targets.nodeid)
        feature_indices = pd.Index(MEASUREMENT_COLUMNS).get_indexer(targets.feature)
        time_indices = pd.Index(times).get_indexer(targets.time)
        if min(sensor_indices.min(), feature_indices.min(), time_indices.min()) < 0:
            raise ValueError("Unknown target time, sensor, or feature")
        self.sensors = sensor_indices
        self.features = feature_indices
        self.times = time_indices
        if self.available[time_indices, sensor_indices, feature_indices].any():
            raise ValueError("Target measurement is visible in model inputs")
        # This is the same centered, boundary-clamped window used by the GRU.
        self.starts = np.clip(time_indices - WINDOW_LENGTH // 2, 0, len(times) - WINDOW_LENGTH)
        self.positions = time_indices - self.starts

    def __len__(self):
        return len(self.target_ids)

    def inputs(self, indices):
        time_indices = self.starts[indices, None] + np.arange(WINDOW_LENGTH)
        sensors = self.neighbors[self.sensors[indices]]
        values = self.values[time_indices[:, :, None], sensors[:, None, :]]
        flags = self.available[time_indices[:, :, None], sensors[:, None, :]]
        windows = np.concatenate([values, flags.astype(np.float32)], axis=3)
        # The GRU selects an output at the target step. Supply that location to
        # the MLP explicitly, including windows clamped at either split boundary.
        positions = np.eye(WINDOW_LENGTH, dtype=np.float32)[self.positions[indices]]
        return np.concatenate([windows.reshape(len(indices), -1), positions], axis=1)


def load_windows(root, split, sensor_ids, neighbors, means, stds):
    frame = pd.read_csv(root / "data/masked" / f"{split}_masked.csv")
    # Never read answer values when constructing model inputs, including test.
    targets = pd.read_csv(
        root / "data/masked" / f"{split}_answers.csv",
        usecols=["target_id", "time", "nodeid", "feature"],
    )
    return MaskedWindows(frame, targets, sensor_ids, neighbors, means, stds)
