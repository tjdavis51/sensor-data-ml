"""Describe saved voltage residuals by sensor and elapsed dataset time."""

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/sensor-nn-matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils import evaluation

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "src/models/neural_network/results/voltage_diagnostics"
SOURCES = {
    "val": ROOT / "src/models/neural_network/results/hidden128_64_wd0.0001_epochs60_seed43/val_predictions.csv",
    "test": ROOT / "src/models/neural_network/results/selected128_64_seed43_test/predictions.csv",
}
BIN_SECONDS = 6 * 60 * 60


def summarize(frame, keys, training_std):
    result = frame.groupby(keys, observed=True).agg(
        count=("error", "size"), bias=("error", "mean"),
        mae=("absolute_error", "mean"), sse=("squared_error", "sum"),
        observed_mean=("true_value", "mean"), predicted_mean=("prediction", "mean"),
        start_time=("time", "min"), end_time=("time", "max"),
    ).reset_index()
    result["rmse"] = np.sqrt(result.sse / result["count"])
    result["nrmse"] = result.rmse / training_std
    totals = frame.groupby("split").squared_error.sum()
    result["split_sse_fraction"] = result.sse / result.split.map(totals)
    return result


def plot_errors(sensor, time, cells):
    fig, axes = plt.subplots(3, 1, figsize=(13, 11), constrained_layout=True)
    colors = {"val": "tab:blue", "test": "tab:orange"}
    table = sensor.pivot(index="nodeid", columns="split", values="rmse")
    table.reindex(columns=["val", "test"]).plot.bar(ax=axes[0], width=0.8, color=[colors["val"], colors["test"]])
    axes[0].set(xlabel="Sensor ID", ylabel="Voltage RMSE (V)", title="Selected model voltage errors by sensor")
    for split, group in time.groupby("split"):
        bins = np.arange(group.time_bin.min(), group.time_bin.max() + BIN_SECONDS, BIN_SECONDS)
        complete = group.set_index("time_bin").reindex(bins)
        axes[1].plot(bins / 86400, complete.rmse, marker="o", label=split, color=colors[split])
        axes[2].plot(bins / 86400, complete["count"].fillna(0), marker="o", label=split, color=colors[split])
    axes[1].set(xlabel="Elapsed dataset day (six-hour bin start)", ylabel="Voltage RMSE (V)", title="Voltage reconstruction error over time")
    axes[1].legend()
    axes[2].set(xlabel="Elapsed dataset day (six-hour bin start)", ylabel="Scored voltage targets", title="Target counts, including bins with no scored targets")
    axes[2].legend()
    fig.savefig(OUTPUT / "sensor_and_time_rmse.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(13, 10), constrained_layout=True)
    vmax = cells.rmse.max()
    for ax, split in zip(axes, ["val", "test"]):
        group = cells[cells.split == split]
        table = group.pivot(index="nodeid", columns="time_bin", values="rmse")
        nodes = np.arange(1, 55)
        bins = np.arange(group.time_bin.min(), group.time_bin.max() + BIN_SECONDS, BIN_SECONDS)
        table = table.reindex(index=nodes, columns=bins)
        mesh = ax.pcolormesh(
            np.append(bins, bins[-1] + BIN_SECONDS) / 86400,
            np.arange(0.5, 55.5), np.ma.masked_invalid(table.to_numpy()),
            cmap="magma", vmin=0, vmax=vmax, shading="flat",
        )
        ax.set(title=f"{split} voltage RMSE by sensor and six-hour bin (white = no scored targets)", xlabel="Elapsed dataset day", ylabel="Sensor ID")
    fig.colorbar(mesh, ax=axes, label="RMSE (V), shared color scale")
    fig.savefig(OUTPUT / "sensor_time_heatmap.png", dpi=160)
    plt.close(fig)


def main():
    os.chdir(ROOT)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    training_std = evaluation.get_training_std()["voltage"]
    pieces = []
    for split, source in SOURCES.items():
        answers = pd.read_csv(ROOT / "data/masked" / f"{split}_answers.csv")
        matched = evaluation.validate_predictions(answers, pd.read_csv(source))
        frame = matched[matched.feature == "voltage"].copy()
        frame["split"] = split
        frame["error"] = frame.prediction - frame.true_value
        frame["absolute_error"] = frame.error.abs()
        frame["squared_error"] = frame.error ** 2
        frame["time_bin"] = (frame.time // BIN_SECONDS) * BIN_SECONDS
        pieces.append(frame)
    errors = pd.concat(pieces, ignore_index=True)
    sensor = summarize(errors, ["split", "nodeid"], training_std)
    time = summarize(errors, ["split", "time_bin"], training_std)
    cells = summarize(errors, ["split", "nodeid", "time_bin"], training_std)
    overall = summarize(errors, ["split"], training_std)
    for name, frame in [("by_sensor", sensor), ("by_time", time), ("by_sensor_time", cells), ("overall", overall)]:
        frame.to_csv(OUTPUT / f"{name}.csv", index=False)

    comparison = sensor.pivot(index="nodeid", columns="split", values=["count", "sse", "bias", "rmse", "nrmse", "observed_mean", "predicted_mean"])
    comparison.columns = [f"{metric}_{split}" for metric, split in comparison.columns]
    for split in ["val", "test"]:
        comparison[f"mse_contribution_{split}"] = comparison[f"sse_{split}"].fillna(0) / len(errors[errors.split == split])
    comparison["mse_gap_contribution"] = comparison.mse_contribution_test - comparison.mse_contribution_val
    comparison.sort_values("mse_gap_contribution", ascending=False).to_csv(OUTPUT / "sensor_gap_contributions.csv")

    distributions = []
    for split in ["train", "val", "test"]:
        frame = pd.read_csv(ROOT / "data/processed" / f"{split}.csv", usecols=["time", "nodeid", "voltage"])
        summary = frame.groupby("nodeid").voltage.agg(["count", "mean", "std", "min", "max"]).reset_index()
        summary["split"] = split
        distributions.append(summary)
    distribution_table = pd.concat(distributions)
    distribution_table.to_csv(OUTPUT / "observed_voltage_by_sensor.csv", index=False)
    training_ranges = distribution_table[distribution_table.split == "train"].set_index("nodeid")
    errors["sensor_training_min"] = errors.nodeid.map(training_ranges["min"])
    errors["sensor_training_max"] = errors.nodeid.map(training_ranges["max"])
    errors["below_sensor_training_min"] = errors.true_value < errors.sensor_training_min
    errors["above_sensor_training_max"] = errors.true_value > errors.sensor_training_max
    errors.to_csv(OUTPUT / "voltage_residuals.csv", index=False)
    summarize(errors, ["split", "below_sensor_training_min"], training_std).to_csv(
        OUTPUT / "training_range_support.csv", index=False
    )
    plot_errors(sensor, time, cells)
    print("Training voltage sample standard deviation", training_std)
    print(overall.to_string(index=False))
    print("Largest validation sensor errors")
    print(sensor[sensor.split == "val"].nlargest(10, "sse").to_string(index=False))
    print("Largest test-minus-validation MSE contributions")
    print(comparison.nlargest(10, "mse_gap_contribution").to_string())


if __name__ == "__main__":
    main()
