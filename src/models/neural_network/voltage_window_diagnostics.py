"""Audit the visible voltage evidence in saved prediction windows."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.models.neural_network.data import WINDOW_LENGTH, load_windows
from src.models.neural_network.neural_network import SensorMLP

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "src/models/neural_network/results/voltage_diagnostics"


def count_visible(values, flags, condition):
    return (flags & condition(values)).sum(axis=1)


def summarize(frame, keys):
    return frame.groupby(keys, observed=True).agg(
        targets=("target_id", "size"), mean_error=("error", "mean"),
        squared_error=("squared_error", "sum"),
        own_visible_fraction=("own_visible", "mean"),
        own_lower_fraction=("own_lower", "mean"),
        own_near_truth_fraction=("own_near_truth", "mean"),
        own_near_truth_both_sides_fraction=("own_near_truth_both_sides", "mean"),
        neighbor_lower_fraction=("neighbor_lower", "mean"),
    ).reset_index()


def main():
    checkpoint = torch.load(
        ROOT / "src/models/neural_network/results/selected128_64_seed43_test/model.pt",
        map_location="cpu", weights_only=True,
    )
    model = SensorMLP(checkpoint["input_size"], checkpoint["hidden_sizes"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    means = checkpoint["means"].numpy()
    stds = checkpoint["stds"].numpy()
    residuals = pd.read_csv(OUTPUT / "voltage_residuals.csv")
    pieces = []
    for split in ["val", "test"]:
        windows = load_windows(ROOT, split, checkpoint["sensor_ids"].numpy(),
                               checkpoint["neighbors"].numpy(), means, stds)
        errors = residuals[residuals.split == split].copy().reset_index(drop=True)
        indices = pd.Index(windows.target_ids).get_indexer(errors.target_id)
        assert (indices >= 0).all() and (windows.features[indices] == 3).all()
        batches = []
        for start in range(0, len(indices), 512):
            selected = indices[start:start + 512]
            rows = errors.iloc[start:start + 512].copy()
            inputs = windows.inputs(selected)
            with torch.inference_mode():
                predictions = model(torch.from_numpy(inputs))[:, 3].numpy() * stds[3] + means[3]
            np.testing.assert_allclose(predictions, rows.prediction, atol=2e-6, rtol=0)
            # Inspect exactly the standardized values and flags passed to the model.
            tensor = inputs[:, :1600].reshape(-1, WINDOW_LENGTH, 5, 8)
            flags = tensor[:, :, :, 7].astype(bool)
            voltages = tensor[:, :, :, 3] * stds[3] + means[3]
            positions = windows.positions[selected]
            assert not flags[np.arange(len(rows)), positions, 0].any()
            offsets = (np.arange(WINDOW_LENGTH)[None, :] - positions[:, None]) * 30
            own = voltages[:, :, 0]
            own_flags = flags[:, :, 0]
            neighbors = voltages[:, :, 1:].reshape(len(rows), -1)
            neighbor_flags = flags[:, :, 1:].reshape(len(rows), -1)
            predicted = rows.prediction.to_numpy()[:, None]
            truth = rows.true_value.to_numpy()[:, None]
            rows["own_visible_count"] = own_flags.sum(axis=1)
            rows["own_lower_count"] = count_visible(own, own_flags, lambda v: v < predicted - 1e-6)
            rows["own_near_truth_count"] = count_visible(own, own_flags, lambda v: np.abs(v - truth) <= .010)
            near_truth = own_flags & (np.abs(own - truth) <= .010)
            rows["own_near_truth_both_sides"] = ((near_truth & (offsets < 0)).any(axis=1) & (near_truth & (offsets > 0)).any(axis=1))
            rows["neighbor_visible_count"] = neighbor_flags.sum(axis=1)
            rows["neighbor_lower_count"] = count_visible(neighbors, neighbor_flags, lambda v: v < predicted - 1e-6)
            nearest_index = np.where(own_flags, np.abs(offsets), np.inf).argmin(axis=1)
            has_own = own_flags.any(axis=1)
            rows["nearest_own_voltage"] = np.where(has_own, own[np.arange(len(rows)), nearest_index], np.nan)
            rows["nearest_own_offset_seconds"] = np.where(has_own, offsets[np.arange(len(rows)), nearest_index], np.nan)
            rows["own_visible"] = rows.own_visible_count > 0
            rows["own_lower"] = rows.own_lower_count > 0
            rows["own_near_truth"] = rows.own_near_truth_count > 0
            rows["neighbor_lower"] = rows.neighbor_lower_count > 0
            batches.append(rows)
        pieces.append(pd.concat(batches, ignore_index=True))
    frame = pd.concat(pieces, ignore_index=True)
    frame.to_csv(OUTPUT / "window_evidence.csv", index=False)
    threshold_tables = []
    for threshold in [0.0, .010, .020, .050]:
        selected = frame[frame.error > threshold]
        summary = summarize(selected, ["split"])
        summary["positive_error_threshold_volts"] = threshold
        threshold_tables.append(summary)
    thresholds = pd.concat(threshold_tables, ignore_index=True)
    thresholds.to_csv(OUTPUT / "window_evidence_thresholds.csv", index=False)
    sensor_summary = summarize(frame[frame.error > .020], ["split", "nodeid"])
    sensor_summary.to_csv(OUTPUT / "window_evidence_sensors_over20mV.csv", index=False)
    print(thresholds.to_string(index=False))
    print(sensor_summary[sensor_summary.nodeid.isin([16,17,18])].to_string(index=False))
    print(frame[(frame.split == "test") & frame.nodeid.isin([16,17])].nlargest(8, "error")[["target_id", "time", "nodeid", "true_value", "prediction", "error", "own_visible_count", "own_lower_count", "own_near_truth_count", "nearest_own_voltage", "nearest_own_offset_seconds", "neighbor_visible_count"]].to_string(index=False))


if __name__ == "__main__":
    main()
