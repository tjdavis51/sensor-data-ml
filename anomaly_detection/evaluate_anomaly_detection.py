"""Summarize detector coverage and evaluate labeled injections and controls."""

import argparse
from pathlib import Path

import pandas as pd

DETECTIONS_PATH = Path(__file__).resolve().parent / "results/detections.csv"
LABELS_PATH = None
EXAMPLES_PER_LAYER = 3

READING_KEY = ["time", "nodeid", "feature"]
DETECTORS = {
    "rules": "rule_flag",
    "residuals": "residual_flag",
    "combined": "is_anomaly",
}


def read_table(path, columns):
    """Load a nonempty table with unique, complete reading keys."""
    frame = pd.read_csv(path, usecols=columns)
    if frame.empty:
        raise ValueError(f"No rows in {path}")
    if frame[READING_KEY].isna().any().any() or frame.duplicated(READING_KEY).any():
        raise ValueError(f"Reading keys must be complete and unique in {path}")

    return frame


def read_flag(values):
    flags = values.astype(str).str.strip().str.lower().map({
        "true": True, "false": False, "1": True, "0": False,
    })
    if flags.isna().any():
        raise ValueError(f"{values.name} must contain true/false or 1/0")

    return flags.astype(bool)


def load_detections(path):
    detections = read_table(path, READING_KEY + [
        "observed_value", "predicted_value", "anomaly_score", "threshold",
        "score_status", "invalid_reason", "heuristic_reason", "residual_flag", "is_anomaly",
    ])
    return prepare_detections(detections)


def prepare_detections(detections):
    """Validate statuses and flags before computing summary counts."""
    detections = detections.copy()
    if not detections["score_status"].isin(["missing", "invalid", "scored"]).all():
        raise ValueError("Unknown detection score status")

    detections["missing"] = detections["score_status"].eq("missing")
    detections["invalid"] = detections["score_status"].eq("invalid")
    detections["scored"] = detections["score_status"].eq("scored")
    detections["heuristic_flag"] = detections["heuristic_reason"].fillna("").ne("")
    detections["rule_flag"] = detections["invalid"] | detections["heuristic_flag"]
    detections["residual_flag"] = read_flag(detections["residual_flag"])
    detections["is_anomaly"] = read_flag(detections["is_anomaly"])
    detections["heuristic_and_residual"] = detections["heuristic_flag"] & detections["residual_flag"]

    if (detections["residual_flag"] & ~detections["scored"]).any():
        raise ValueError("Only scored readings can have a residual flag")
    if (detections["invalid"] != detections["invalid_reason"].fillna("").ne("")).any():
        raise ValueError("Invalid status and physical rule reasons disagree")
    combined = detections["rule_flag"] | detections["residual_flag"]
    if (detections["is_anomaly"] != combined).any():
        raise ValueError("Combined flags disagree with the rule and residual flags")
    if (detections["missing"] & combined).any():
        raise ValueError("Missing readings cannot be flagged")

    return detections


def summarize(detections, group_column):
    """Use observed readings for rule rates and scored readings for residual rates."""
    groups = detections.groupby(group_column)
    counts = groups[[
        "missing", "invalid", "scored", "heuristic_flag", "residual_flag",
        "heuristic_and_residual", "is_anomaly",
    ]].sum()
    counts.insert(0, "readings", groups.size())
    observed = (counts["readings"] - counts["missing"]).replace(0, float("nan"))
    scored = counts["scored"].replace(0, float("nan"))

    counts["invalid_pct_observed"] = 100 * counts["invalid"] / observed
    counts["heuristic_pct_observed"] = 100 * counts["heuristic_flag"] / observed
    counts["residual_pct_scored"] = 100 * counts["residual_flag"] / scored
    counts["combined_pct_observed"] = 100 * counts["is_anomaly"] / observed
    return counts


def print_examples(detections, limit):
    if limit == 0:
        return

    columns = READING_KEY + ["observed_value", "predicted_value", "anomaly_score", "threshold"]
    layers = {
        "Physical rule": ("invalid", ["invalid_reason"]),
        "Heuristic rule": ("heuristic_flag", ["heuristic_reason"]),
        "Residual": ("residual_flag", []),
    }
    for name, (flag, reasons) in layers.items():
        examples = detections.loc[detections[flag]].sort_values(READING_KEY).head(limit)
        print(f"\n{name} examples")
        if examples.empty:
            print("No flagged readings")
        else:
            print(examples[columns + reasons].to_string(index=False))


def evaluate_labels(detections, labels_path):
    """Evaluate only explicitly labeled injections and selected untouched controls."""
    labels = read_table(labels_path, READING_KEY + ["is_injected", "anomaly_type"])
    labels["is_injected"] = read_flag(labels["is_injected"])
    labels["anomaly_type"] = labels["anomaly_type"].fillna("").astype(str).str.strip()
    if (labels["is_injected"] & labels["anomaly_type"].eq("")).any():
        raise ValueError("Each injected reading needs an anomaly_type")

    matched = labels.merge(
        detections, on=READING_KEY, how="left", validate="one_to_one", indicator=True
    )
    unmatched = matched["_merge"].ne("both").sum()
    if unmatched:
        raise ValueError(f"{unmatched:,} labeled readings are absent from the detections")
    if matched["missing"].any():
        raise ValueError("Labels must refer to observed readings")

    positives = matched[matched["is_injected"]]
    controls = matched[~matched["is_injected"]]
    rows = []
    for name, flag in DETECTORS.items():
        true_positives = int(positives[flag].sum())
        false_positives = int(controls[flag].sum())
        alerts = true_positives + false_positives
        rows.append({
            "detector": name,
            "detected": true_positives,
            "missed": len(positives) - true_positives,
            "control_alerts": false_positives,
            "controls": len(controls),
            "detection_pct": 100 * true_positives / len(positives) if len(positives) else None,
            "control_alert_pct": 100 * false_positives / len(controls) if len(controls) else None,
            "precision_pct": 100 * true_positives / alerts if alerts and len(controls) else None,
            "f1_pct": (200 * true_positives / (len(positives) + alerts)
                       if len(positives) and len(controls) else None),
        })

    metrics = pd.DataFrame(rows)
    print("\nLabeled comparison")
    print(metrics.to_string(index=False, float_format="{:.2f}".format, na_rep="-"))
    if not positives.empty:
        groups = positives.groupby("anomaly_type")
        rates = 100 * groups[list(DETECTORS.values())].mean()
        rates.columns = [f"{name}_detected_pct" for name in DETECTORS]
        rates.insert(0, "injected", groups.size())
        print("\nDetection by injected anomaly type")
        print(rates.to_string(float_format="{:.2f}".format))

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Summarize GRU anomaly detector results")
    parser.add_argument("--detections", type=Path, default=DETECTIONS_PATH, help="detection CSV path")
    parser.add_argument("--labels", type=Path, default=LABELS_PATH,
                        help="optional CSV of injected readings and controls")
    parser.add_argument("--examples", type=int, default=EXAMPLES_PER_LAYER,
                        help="examples to show per detection layer")
    args = parser.parse_args()
    if args.examples < 0:
        parser.error("--examples must be nonnegative")

    detections = load_detections(args.detections)
    print(f"Loaded {len(detections):,} readings from {args.detections}")
    for column in ["feature", "nodeid"]:
        print(f"\nCoverage and flag rates by {column}")
        print(summarize(detections, column).to_string(float_format="{:.2f}".format, na_rep="-"))
    print_examples(detections, args.examples)

    if args.labels is not None:
        evaluate_labels(detections, args.labels)


if __name__ == "__main__":
    main()
