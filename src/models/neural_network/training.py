"""Train on fixed shared targets, select on validation, then score held-out test."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.models.gru.gru import (
    build_sensor_arrays,
    find_nearest_neighbors,
    fit_normalization,
)
from src.models.neural_network.data import INPUT_SIZE, load_windows
from src.models.neural_network.neural_network import SensorMLP
from src.models.neural_network.voltage_reference import build_reference, fit_fallbacks
from src.utils import evaluation

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "src/models/neural_network/results"


def save_csv(frame, path):
    temporary = path.with_suffix(".csv.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def predict(model, windows, batch_size):
    model.eval()
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(windows), batch_size):
            indices = np.arange(start, min(start + batch_size, len(windows)))
            values = model(torch.from_numpy(windows.inputs(indices)))
            selected = values[torch.arange(len(indices)), windows.features[indices]]
            outputs.append(selected.numpy())
    result = np.concatenate(outputs)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite model prediction")
    return result


def normalized_labels(split, features, means, stds):
    answers = pd.read_csv(ROOT / "data/masked" / f"{split}_answers.csv")
    labels = (answers.true_value.to_numpy() - means[features]) / stds[features]
    if not np.isfinite(labels).all():
        raise ValueError("Nonfinite training or validation label")
    return labels.astype(np.float32)


def train(args, output_dir=OUTPUT):
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    frame = pd.read_csv(ROOT / "data/processed/train.csv")
    raw, _, _, sensor_ids = build_sensor_arrays(frame)
    neighbors = find_nearest_neighbors(frame, sensor_ids, 4)
    means, stds = fit_normalization(raw)
    if not np.isfinite(means).all() or not np.isfinite(stds).all():
        raise ValueError("Training statistics must be finite for all measurements")
    training = load_windows(ROOT, "train", sensor_ids, neighbors, means, stds)
    validation = load_windows(ROOT, "val", sensor_ids, neighbors, means, stds)
    labels = normalized_labels("train", training.features, means, stds)
    val_labels = normalized_labels("val", validation.features, means, stds)
    fallbacks = None
    if args.voltage_reference:
        fallbacks = fit_fallbacks(raw)
        train_reference, _ = build_reference(training, fallbacks, means, stds)
        val_reference, _ = build_reference(validation, fallbacks, means, stds)
        labels -= train_reference
        val_labels -= val_reference
    model = SensorMLP(INPUT_SIZE, args.hidden_sizes)
    if args.voltage_reference:
        # Start the voltage prediction exactly at the reference.
        with torch.no_grad():
            model.network[-1].weight[3].zero_()
            model.network[-1].bias[3].zero_()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = None
    if args.lr_patience:
        # PyTorch reduces after patience + 1 bad epochs. Subtract one to honor
        # the CLI's literal count, using the same strict improvement as selection.
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=args.lr_patience - 1,
            threshold=0.0, threshold_mode="abs", eps=0.0,
        )
    best_loss = float("inf")
    history = []
    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        learning_rate = optimizer.param_groups[0]["lr"]
        model.train()
        total_error = 0.0
        order = rng.permutation(len(training))
        for start in range(0, len(order), args.batch_size):
            indices = order[start:start + args.batch_size]
            inputs = torch.from_numpy(training.inputs(indices))
            expected = torch.from_numpy(labels[indices])
            optimizer.zero_grad()
            outputs = model(inputs)
            selected = outputs[torch.arange(len(indices)), training.features[indices]]
            loss = torch.mean((selected - expected) ** 2)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            optimizer.step()
            total_error += loss.item() * len(indices)
        val_errors = (predict(model, validation, args.batch_size) - val_labels) ** 2
        val_all_loss = float(np.mean(val_errors))
        selection_mask = validation.features == 3 if args.voltage_reference else np.ones(len(validation), dtype=bool)
        val_loss = float(np.mean(val_errors[selection_mask]))
        history.append({"epoch": epoch, "train_mse": total_error / len(training),
                        "val_mse": val_loss, "val_all_mse": val_all_loss, "learning_rate": learning_rate})
        print(f"Epoch {epoch}: train MSE {history[-1]['train_mse']:.6f}, val MSE {val_loss:.6f}, LR {learning_rate:.8g}", flush=True)
        if val_loss < best_loss:
            best_loss = val_loss
            stale_epochs = 0
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "means": torch.from_numpy(means), "stds": torch.from_numpy(stds),
                "sensor_ids": torch.from_numpy(sensor_ids),
                "neighbors": torch.from_numpy(neighbors),
                "input_size": INPUT_SIZE, "epoch": epoch, "val_mse": val_loss,
                "hidden_sizes": args.hidden_sizes,
                "seed": args.seed,
            }
            if fallbacks is not None:
                checkpoint["voltage_fallbacks"] = torch.from_numpy(fallbacks)
            torch.save(checkpoint, output_dir / "model.pt.tmp")
            (output_dir / "model.pt.tmp").replace(output_dir / "model.pt")
        else:
            stale_epochs += 1
        if scheduler is not None:
            scheduler.step(val_loss)
        history[-1]["next_learning_rate"] = optimizer.param_groups[0]["lr"]
        save_csv(pd.DataFrame(history), output_dir / "training.csv")
        if stale_epochs >= args.patience:
            break
    metadata = {
        "arguments": vars(args), "best_val_mse": best_loss,
        "epochs_completed": len(history), "input_size": INPUT_SIZE,
        "architecture": [INPUT_SIZE, *args.hidden_sizes, 4],
        "torch_version": torch.__version__, "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "selection": ("minimum voltage validation MSE in training-standardized units"
                      if args.voltage_reference else "minimum validation MSE in training-standardized units"),
        "voltage_reference": ("masked-window same-sensor linear interpolation, one-sided nearest, training sensor median fallback"
                              if args.voltage_reference else None),
        "scheduler": {"name": "ReduceLROnPlateau", "factor": 0.5,
                      "non_improving_epochs": args.lr_patience, "threshold": 0.0}
                     if args.lr_patience else None,
        "shared_mask": json.loads((ROOT / "data/masked/masking_config.json").read_text()),
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def evaluate(split, batch_size, output_dir=OUTPUT):
    checkpoint = torch.load(output_dir / "model.pt", map_location="cpu", weights_only=True)
    model = SensorMLP(checkpoint["input_size"], checkpoint.get("hidden_sizes", [64]))
    model.load_state_dict(checkpoint["model_state_dict"])
    means = checkpoint["means"].numpy()
    stds = checkpoint["stds"].numpy()
    windows = load_windows(ROOT, split, checkpoint["sensor_ids"].numpy(),
                           checkpoint["neighbors"].numpy(), means, stds)
    predictions = predict(model, windows, batch_size)
    if "voltage_fallbacks" in checkpoint:
        reference, _ = build_reference(windows, checkpoint["voltage_fallbacks"].numpy(), means, stds)
        predictions += reference
    predictions = predictions * stds[windows.features] + means[windows.features]
    result = pd.DataFrame({"target_id": windows.target_ids, "prediction": predictions})
    filename = "predictions.csv" if split == "test" else "val_predictions.csv"
    save_csv(result, output_dir / filename)
    # Read final labels only after all predictions have been generated and saved.
    answers = pd.read_csv(ROOT / "data/masked" / f"{split}_answers.csv")
    matched = evaluation.validate_predictions(answers, result)
    metrics, score = evaluation.evaluate_predictions(matched, evaluation.get_training_std())
    evaluation.print_results("neural_network", split, metrics, score)
    if split == "test":
        evaluation.save_results(metrics, score, output_dir)
    else:
        save_csv(metrics, output_dir / "val_evaluation.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["train", "test", "both"], default="both")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--output-dir", default=str(OUTPUT))
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[64])
    parser.add_argument("--voltage-reference", action="store_true",
                        help="Predict voltage correction to a masked-window reference and select on voltage validation MSE")
    parser.add_argument("--lr-patience", type=int, default=0,
                        help="Halve LR after this many non-improving validation epochs (0 disables)")
    args = parser.parse_args()
    if args.lr_patience < 0:
        parser.error("LR patience must be nonnegative")
    if min(args.hidden_sizes) < 1:
        parser.error("Hidden layer sizes must be positive")
    if min(args.epochs, args.batch_size, args.patience) < 1 or not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("Epochs, batch size, patience, and learning rate must be positive")
    if not np.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("Weight decay must be finite and nonnegative")
    os.chdir(ROOT)  # Shared evaluation paths are relative to the repository root.
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode in ("train", "both"):
        train(args, output_dir)
        evaluate("val", args.batch_size, output_dir)
    if args.mode in ("test", "both"):
        evaluate("test", args.batch_size, output_dir)


if __name__ == "__main__":
    main()
