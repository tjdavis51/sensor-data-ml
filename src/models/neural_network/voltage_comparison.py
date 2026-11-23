"""Compare voltage reference, direct MLP, and corrected MLP by window support."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.models.gru.gru import build_sensor_arrays
from src.models.neural_network.data import load_windows
from src.models.neural_network.voltage_reference import build_reference, fit_fallbacks
from src.utils import evaluation

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = ROOT / "src/models/neural_network/results/voltage_correction_seed43"
ORIGINAL = ROOT / "src/models/neural_network/results/selected128_64_seed43_test/model.pt"


def compare(split, output):
    checkpoint = torch.load(output / "model.pt" if (output / "model.pt").exists() else ORIGINAL,
                            map_location="cpu", weights_only=True)
    means = checkpoint["means"].numpy()
    stds = checkpoint["stds"].numpy()
    sensors = checkpoint["sensor_ids"].numpy()
    if "voltage_fallbacks" in checkpoint:
        fallback = checkpoint["voltage_fallbacks"].numpy()
    else:
        frame = pd.read_csv(ROOT / "data/processed/train.csv")
        values, _, _, _ = build_sensor_arrays(frame, sensors)
        fallback = fit_fallbacks(values)
    windows = load_windows(ROOT, split, sensors, checkpoint["neighbors"].numpy(), means, stds)
    reference, count = build_reference(windows, fallback, means, stds)
    indices = np.flatnonzero(windows.features == 3)
    reference_frame = pd.DataFrame({"target_id": windows.target_ids[indices],
                                    "prediction": reference[indices] * stds[3] + means[3],
                                    "visible_count": count[indices]})
    reference_frame.to_csv(output / f"{split}_voltage_reference.csv", index=False)
    all_answers = pd.read_csv(ROOT / "data/masked" / f"{split}_answers.csv")
    answers = all_answers[all_answers.feature == "voltage"]
    original_path = (ROOT / "src/models/neural_network/results/hidden128_64_wd0.0001_epochs60_seed43/val_predictions.csv"
                     if split == "val" else ROOT / "src/models/neural_network/results/selected128_64_seed43_test/predictions.csv")
    original = pd.read_csv(original_path)
    evaluation.validate_predictions(all_answers, original)
    methods = {"reference": reference_frame[["target_id", "prediction"]], "current_mlp": original}
    corrected_path = output / ("val_predictions.csv" if split == "val" else "predictions.csv")
    if corrected_path.exists():
        corrected = pd.read_csv(corrected_path)
        evaluation.validate_predictions(all_answers, corrected)
        methods["corrected_mlp"] = corrected
    support = reference_frame.set_index("target_id").visible_count
    direct_errors = answers.merge(original, on="target_id", validate="one_to_one").set_index("target_id")
    sparse23 = direct_errors.index[((direct_errors.prediction - direct_errors.true_value) > .020) & (direct_errors.index.map(support) == 0)]
    training_std = pd.read_csv(ROOT / "data/processed/train.csv", usecols=["voltage"]).voltage.std()
    metrics = []
    for method, predictions in methods.items():
        selected = predictions[predictions.target_id.isin(answers.target_id)]
        matched = evaluation.validate_predictions(answers, selected)
        has_own = matched.target_id.map(support) > 0
        groups = {"all": np.ones(len(matched), dtype=bool), "with_same_sensor": has_own,
                  "without_same_sensor": ~has_own}
        if split == "test":
            groups["original_sparse_over20mV"] = matched.target_id.isin(sparse23)
        for group, mask in groups.items():
            rows = matched[mask]
            error = rows.prediction - rows.true_value
            mse = np.mean(error ** 2)
            metrics.append({"split": split, "method": method, "group": group, "count": len(rows),
                            "bias": error.mean(), "mae": np.mean(np.abs(error)), "rmse": np.sqrt(mse),
                            "nrmse": np.sqrt(mse) / training_std, "sse": np.sum(error ** 2)})
    table = pd.DataFrame(metrics)
    table.to_csv(output / f"{split}_voltage_comparison.csv", index=False)
    print(table.to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    compare(args.split, args.output_dir)


if __name__ == "__main__":
    main()
