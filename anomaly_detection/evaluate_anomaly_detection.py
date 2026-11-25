# evaluation for the gru anomaly detector

# input from the detection csv with optional experiment labels
# coverage and alert rates grouped by measurement feature and sensor
# examples of physical rule heuristic rule and residual alerts
# labeled comparison of rules alone residuals alone and the combined detector
# run settings at the top of this file and optional overrides from the command line

# load the helpers for run settings and file paths
import argparse
from pathlib import Path

# use pandas to join and summarize the detection tables
import pandas as pd

# default run settings for python anomaly_detection/evaluate_anomaly_detection.py
DETECTIONS_PATH = Path(__file__).resolve().parent / "results/detections.csv"
# leave labels unset for a descriptive summary of real data
LABELS_PATH = None
# show a few examples from each detection layer by default
EXAMPLES_PER_LAYER = 3

# identify each measurement within the detectors and labels
READING_KEY = ["time", "nodeid", "feature"]
# match each comparison name to its flag column
DETECTORS = {
    "rules": "rule_flag",
    "residuals": "residual_flag",
    "combined": "is_anomaly",
}


# read and validate a table keyed by individual measurements
def read_table(path, columns):
    # load only the columns needed for the evaluation
    frame = pd.read_csv(path, usecols=columns)
    # stop if there are no readings to summarize
    if frame.empty:
        raise ValueError(f"No rows in {path}")
    # require each key to identify exactly one reading
    if frame[READING_KEY].isna().any().any() or frame.duplicated(READING_KEY).any():
        raise ValueError(f"Reading keys must be complete and unique in {path}")

    # return the validated table
    return frame


# convert flags from csv text to booleans
def read_flag(values):
    # accept the detector's boolean format and numeric labels
    flags = values.astype(str).str.strip().str.lower().map({
        "true": True, "false": False, "1": True, "0": False,
    })
    # reject values outside the accepted flag formats
    if flags.isna().any():
        raise ValueError(f"{values.name} must contain true/false or 1/0")

    # return booleans for counting and comparing flags
    return flags.astype(bool)


# check the detector output and build the flags used in summaries
def load_detections(path):
    # load the keys values reasons and flag columns
    detections = read_table(path, READING_KEY + [
        "observed_value", "predicted_value", "anomaly_score", "threshold",
        "score_status", "invalid_reason", "heuristic_reason", "residual_flag", "is_anomaly",
    ])
    return prepare_detections(detections)


# use the same checks and flags for saved and in-memory detections
def prepare_detections(detections):
    detections = detections.copy()
    # require each reading to have a recognized scoring status
    if not detections["score_status"].isin(["missing", "invalid", "scored"]).all():
        raise ValueError("Unknown detection score status")

    # mark every reading with no observed value
    detections["missing"] = detections["score_status"].eq("missing")
    # mark readings rejected by a physical rule
    detections["invalid"] = detections["score_status"].eq("invalid")
    # mark readings with an eligible residual score
    detections["scored"] = detections["score_status"].eq("scored")
    # treat a nonempty heuristic reason as a heuristic alert
    detections["heuristic_flag"] = detections["heuristic_reason"].fillna("").ne("")
    # combine the physical and heuristic signals into the rules-only flag
    detections["rule_flag"] = detections["invalid"] | detections["heuristic_flag"]
    # read the residual flags as booleans
    detections["residual_flag"] = read_flag(detections["residual_flag"])
    # read the overall flags as booleans
    detections["is_anomaly"] = read_flag(detections["is_anomaly"])
    # count heuristic and residual overlap separately
    detections["heuristic_and_residual"] = detections["heuristic_flag"] & detections["residual_flag"]

    # confirm that skipped readings never have a residual flag
    if (detections["residual_flag"] & ~detections["scored"]).any():
        raise ValueError("Only scored readings can have a residual flag")
    # confirm that each invalid status has a physical rule reason
    if (detections["invalid"] != detections["invalid_reason"].fillna("").ne("")).any():
        raise ValueError("Invalid status and physical rule reasons disagree")
    # reconstruct the expected combined flag from its components
    combined = detections["rule_flag"] | detections["residual_flag"]
    # compare the expected flag with the saved combined flag
    if (detections["is_anomaly"] != combined).any():
        raise ValueError("Combined flags disagree with the rule and residual flags")
    # keep missing readings out of all alert counts
    if (detections["missing"] & combined).any():
        raise ValueError("Missing readings cannot be flagged")

    # return the checked readings and derived flags
    return detections


# summarize counts and rates by feature or sensor
def summarize(detections, group_column):
    # group readings by the requested column
    groups = detections.groupby(group_column)
    # count the statuses and flags within each group
    counts = groups[[
        "missing", "invalid", "scored", "heuristic_flag", "residual_flag",
        "heuristic_and_residual", "is_anomaly",
    ]].sum()
    # include the total rows in each group for coverage
    counts.insert(0, "readings", groups.size())
    # use observed readings for physical heuristic and combined rates
    observed = (counts["readings"] - counts["missing"]).replace(0, float("nan"))
    # use only scored readings for the residual rate
    scored = counts["scored"].replace(0, float("nan"))

    # use only readings eligible for each type of flag as the denominator
    counts["invalid_pct_observed"] = 100 * counts["invalid"] / observed
    # report the share of observed readings that trigger a heuristic
    counts["heuristic_pct_observed"] = 100 * counts["heuristic_flag"] / observed
    # report the share of scored readings above their residual threshold
    counts["residual_pct_scored"] = 100 * counts["residual_flag"] / scored
    # report the share of observed readings flagged by any layer
    counts["combined_pct_observed"] = 100 * counts["is_anomaly"] / observed
    # return the grouped counts and percentages
    return counts


# print a few flagged readings from each layer
def print_examples(detections, limit):
    # skip the examples when their count is set to zero
    if limit == 0:
        return

    # include enough values to explain a residual alert
    columns = READING_KEY + ["observed_value", "predicted_value", "anomaly_score", "threshold"]
    # choose the flag and reason columns for each layer
    layers = {
        "Physical rule": ("invalid", ["invalid_reason"]),
        "Heuristic rule": ("heuristic_flag", ["heuristic_reason"]),
        "Residual": ("residual_flag", []),
    }
    # show the layers independently so an overlapping reading can appear twice
    for name, (flag, reasons) in layers.items():
        # pick the earliest flagged readings in a consistent order
        examples = detections.loc[detections[flag]].sort_values(READING_KEY).head(limit)
        # identify the layer before printing its examples
        print(f"\n{name} examples")
        # handle layers with no alerts in this input
        if examples.empty:
            print("No flagged readings")
        else:
            # print the selected readings and any rule reasons
            print(examples[columns + reasons].to_string(index=False))


# compare detector flags against known injected cases and chosen controls
def evaluate_labels(detections, labels_path):
    # use only explicitly labeled positives and controls
    labels = read_table(labels_path, READING_KEY + ["is_injected", "anomaly_type"])
    # convert the injected and control labels to booleans
    labels["is_injected"] = read_flag(labels["is_injected"])
    # allow an empty anomaly type for controls
    labels["anomaly_type"] = labels["anomaly_type"].fillna("").astype(str).str.strip()
    # require a type for every injected case
    if (labels["is_injected"] & labels["anomaly_type"].eq("")).any():
        raise ValueError("Each injected reading needs an anomaly_type")

    # join the labels to detections using the complete reading key
    matched = labels.merge(
        detections, on=READING_KEY, how="left", validate="one_to_one", indicator=True
    )
    # count any labels with no corresponding result
    unmatched = matched["_merge"].ne("both").sum()
    # fail rather than leaving unmatched labels out of the comparison
    if unmatched:
        raise ValueError(f"{unmatched:,} labeled readings are absent from the detections")
    # require a real observed value for every labeled case
    if matched["missing"].any():
        raise ValueError("Labels must refer to observed readings")

    # collect readings changed by the injector
    positives = matched[matched["is_injected"]]
    # collect explicitly labeled untouched readings
    controls = matched[~matched["is_injected"]]
    # collect one comparison row for each detector
    rows = []
    # evaluate the same labeled readings for each flag combination
    for name, flag in DETECTORS.items():
        # count injected readings that were flagged
        true_positives = int(positives[flag].sum())
        # count alerts on the selected controls
        false_positives = int(controls[flag].sum())
        # count all alerts in the labeled sample
        alerts = true_positives + false_positives
        # calculate detection and control rates with their own denominators
        rows.append({
            "detector": name,
            "detected": true_positives,
            "missed": len(positives) - true_positives,
            "control_alerts": false_positives,
            "controls": len(controls),
            "detection_pct": 100 * true_positives / len(positives) if len(positives) else None,
            "control_alert_pct": 100 * false_positives / len(controls) if len(controls) else None,
            "precision_pct": 100 * true_positives / alerts if alerts and len(controls) else None,
        })

    # print the results for the labeled sample
    print("\nLabeled comparison")
    print(pd.DataFrame(rows).to_string(index=False, float_format="{:.2f}".format, na_rep="-"))
    # show detection rates for individual injection types when positives exist
    if not positives.empty:
        # group the injected readings by anomaly type
        groups = positives.groupby("anomaly_type")
        # calculate the share detected by each layer within each type
        rates = 100 * groups[list(DETECTORS.values())].mean()
        # label the percentage columns by detector
        rates.columns = [f"{name}_detected_pct" for name in DETECTORS]
        # include counts to show how many readings contributed to each rate
        rates.insert(0, "injected", groups.size())
        # identify this table as a breakdown of the injected cases
        print("\nDetection by injected anomaly type")
        print(rates.to_string(float_format="{:.2f}".format))


# read run settings and print the requested evaluation
def main():
    # set up the optional overrides for the constants above
    parser = argparse.ArgumentParser(description="Summarize GRU anomaly detector results")
    # allow a different detection file for the summary
    parser.add_argument("--detections", type=Path, default=DETECTIONS_PATH, help="detection CSV path")
    # allow a labels file for a controlled comparison
    parser.add_argument("--labels", type=Path, default=LABELS_PATH,
                        help="optional CSV of injected readings and controls")
    # allow the number of alert examples to be adjusted
    parser.add_argument("--examples", type=int, default=EXAMPLES_PER_LAYER,
                        help="examples to show per detection layer")
    # read the selected values and defaults
    args = parser.parse_args()
    # reject negative example counts
    if args.examples < 0:
        parser.error("--examples must be nonnegative")

    # load and validate the detector output
    detections = load_detections(args.detections)
    # show the input file and how many rows it contains
    print(f"Loaded {len(detections):,} readings from {args.detections}")
    # summarize the data by feature and by sensor
    for column in ["feature", "nodeid"]:
        # identify the grouping used for the next table
        print(f"\nCoverage and flag rates by {column}")
        # print the counts and percentages for each group
        print(summarize(detections, column).to_string(float_format="{:.2f}".format, na_rep="-"))
    # show examples after the aggregate summaries
    print_examples(detections, args.examples)

    # calculate labeled metrics only when a labels file is supplied
    if args.labels is not None:
        evaluate_labels(detections, args.labels)


# run the entry point when this file is started directly
if __name__ == "__main__":
    main()
