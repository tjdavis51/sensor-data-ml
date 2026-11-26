"""Compare original and injected held-out readings with a frozen GRU detector."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from anomaly_detection.anomaly_detection import (
    INPUT_COLUMNS,
    PROCESSED_DIR,
    TIME_STEP,
    WINDOW_LENGTH,
    load_checkpoint,
    prepare_inputs,
    quality_rules,
    reading_rows,
    score_readings,
)
from anomaly_detection.evaluate_anomaly_detection import (
    DETECTORS,
    READING_KEY,
    evaluate_labels,
    prepare_detections,
)


START_TIME = 1423560
END_TIME = 1728000
OUTPUT_DIR = ROOT / "anomaly_detection/results/injection"
SEED = 42
EPISODES_PER_TYPE = 8
CONTROLS_PER_INJECTED_READING = 2

# (type, feature, duration in 30-second steps, offset or replacement value)
# Stuck humidity repeats the first reading; that unchanged anchor is not labeled.
INJECTIONS = [
    ("stuck_humidity", "humidity", 10, None),
    ("temperature_offset", "temperature", 10, 4.0),
    ("light_spike", "light", 1, 400.0),
    ("invalid_humidity", "humidity", 1, -5.0),
]
MIN_STUCK_CHANGE = 0.3
MAX_INJECTED_LIGHT = 1800.0


def load_experiment_input():
    """Load original test observations, retaining context within the test split."""
    test = pd.read_csv(PROCESSED_DIR / "test.csv", usecols=["time", "nodeid"])
    test_start = test["time"].min()
    test_end = test["time"].max() + TIME_STEP
    if not test_start <= START_TIME < END_TIME <= test_end:
        raise ValueError("The injection interval must be within the test split")

    padding = WINDOW_LENGTH * TIME_STEP
    context_start = max(test_start, START_TIME - padding)
    context_end = min(test_end, END_TIME + padding)
    source = pd.read_csv(PROCESSED_DIR / "sensor_data.csv", usecols=INPUT_COLUMNS)
    context = source[source["time"].ge(context_start) & source["time"].lt(context_end)]
    return context.sort_values(["time", "nodeid"]).reset_index(drop=True)


def choose_injections(original, rng):
    """Sample separated episodes and label only cells whose values actually change."""
    invalid, heuristic = quality_rules(original)
    modified = original.copy()
    within_slice = original["time"].ge(START_TIME) & original["time"].lt(END_TIME)
    used_times = []
    labels = []

    for anomaly_type, feature, length, amount in INJECTIONS:
        usable = original[within_slice & original[feature].notna()
                          & invalid[feature].eq("") & heuristic[feature].eq("")]
        candidates = []
        for _, sensor in usable.groupby("nodeid"):
            times = sensor["time"].to_numpy()
            values = sensor[feature].to_numpy()
            indices = sensor.index.to_numpy()
            for start in range(len(sensor) - length + 1):
                stop = start + length
                if times[stop - 1] - times[start] != (length - 1) * TIME_STEP:
                    continue

                current = values[start:stop]
                if anomaly_type == "stuck_humidity":
                    if np.max(np.abs(current - current[0])) < MIN_STUCK_CHANGE:
                        continue
                    injected = np.full(length, current[0])
                elif anomaly_type == "invalid_humidity":
                    injected = np.full(length, amount)
                else:
                    injected = current + amount
                    if feature == "temperature" and (injected > 60).any():
                        continue
                    if feature == "light" and (injected > MAX_INJECTED_LIGHT).any():
                        continue

                candidates.append((indices[start:stop], injected, times[start], times[stop - 1]))

        count = 0
        for candidate in rng.permutation(len(candidates)):
            indices, injected, first_time, last_time = candidates[candidate]

            padding = WINDOW_LENGTH * TIME_STEP
            if any(first_time <= end + padding and last_time >= start - padding
                   for start, end in used_times):
                continue

            count += 1
            episode_id = f"{anomaly_type}_{count:02d}"
            used_times.append((first_time, last_time))
            for index, value in zip(indices, injected):
                before = original.at[index, feature]
                if value == before:
                    continue

                modified.at[index, feature] = value
                labels.append({
                    "time": original.at[index, "time"],
                    "nodeid": original.at[index, "nodeid"],
                    "feature": feature,
                    "is_injected": True,
                    "anomaly_type": anomaly_type,
                    "episode_id": episode_id,
                    "original_value": before,
                    "experiment_value": value,
                })

            if count == EPISODES_PER_TYPE:
                break

        if count != EPISODES_PER_TYPE:
            raise ValueError(
                f"Placed {count} of {EPISODES_PER_TYPE} {anomaly_type} episodes "
                "with the required separation. Increase the interval or lower the episode count."
            )

    return modified, pd.DataFrame(labels), used_times


def choose_controls(original, injected_labels, used_times, rng):
    """Sample rule-passing controls outside every injection window."""
    invalid, heuristic = quality_rules(original)
    eligible_times = original["time"].ge(START_TIME) & original["time"].lt(END_TIME)

    padding = WINDOW_LENGTH * TIME_STEP
    for start, end in used_times:
        eligible_times &= ~original["time"].between(start - padding, end + padding)

    controls = []
    # Controls match the injected feature mix, not the natural anomaly prevalence.
    for feature, count in injected_labels.groupby("feature").size().items():
        eligible = original[eligible_times & original[feature].notna()
                            & invalid[feature].eq("") & heuristic[feature].eq("")]
        sample_size = count * CONTROLS_PER_INJECTED_READING
        if len(eligible) < sample_size:
            raise ValueError(f"Need {sample_size} eligible control readings for {feature}")

        positions = rng.choice(len(eligible), size=sample_size, replace=False)
        for row in eligible.iloc[positions].itertuples(index=False):
            value = getattr(row, feature)
            controls.append({
                "time": row.time,
                "nodeid": row.nodeid,
                "feature": feature,
                "is_injected": False,
                "anomaly_type": "",
                "episode_id": "",
                "original_value": value,
                "experiment_value": value,
            })

    return pd.DataFrame(controls)


def score_experiment(observations, labels, model, means, stds, sensor_ids, neighbors):
    """Score labeled readings using the full original or modified context."""
    invalid, heuristic = quality_rules(observations)
    values, availability, times = prepare_inputs(observations, invalid, means, stds, sensor_ids)
    readings = reading_rows(observations, invalid, heuristic)
    targets = labels[READING_KEY].merge(readings, on=READING_KEY, validate="one_to_one")
    if len(targets) != len(labels):
        raise ValueError("Some experiment targets are absent from the input")

    return score_readings(
        targets, model, values, availability, times, sensor_ids, neighbors, means, stds
    )


def compare_with_baseline(labels, baseline, detections, output_dir):
    """Compare reading and episode alerts before and after injection."""
    flag_columns = list(DETECTORS.values())
    comparison = labels.merge(
        baseline[READING_KEY + flag_columns], on=READING_KEY, validate="one_to_one"
    ).merge(
        detections[READING_KEY + flag_columns], on=READING_KEY, validate="one_to_one",
        suffixes=("_baseline", "_injected")
    )
    if len(comparison) != len(labels):
        raise ValueError("Some labels are missing from the original or modified detections")

    positives = comparison[comparison["is_injected"]].copy()
    groups = positives.groupby(["anomaly_type", "episode_id"])
    episodes = groups.agg(
        time_start=("time", "min"), time_end=("time", "max"),
        nodeid=("nodeid", "first"), feature=("feature", "first"),
        changed_readings=("time", "size"),
    )
    summary = episodes.groupby(level="anomaly_type").size().rename("episodes").to_frame()
    for name, flag in DETECTORS.items():
        before = f"{flag}_baseline"
        after = f"{flag}_injected"
        new = f"{name}_new_alert"
        comparison[new] = comparison[after] & ~comparison[before]
        episodes[f"{name}_baseline"] = groups[before].any()
        episodes[f"{name}_detected"] = groups[after].any()
        episodes[new] = episodes[f"{name}_detected"] & ~episodes[f"{name}_baseline"]
        summary[f"{name}_detected"] = episodes.groupby(level="anomaly_type")[f"{name}_detected"].sum()
        summary[f"{name}_new"] = episodes.groupby(level="anomaly_type")[new].sum()

    comparison.to_csv(output_dir / "reading_comparison.csv", index=False)
    episodes.to_csv(output_dir / "episode_comparison.csv")
    print("\nEpisode comparison")
    print(summary.to_string())
    print("A new episode alert has a flag after injection and no baseline flag on its labeled readings")


def main():
    rng = np.random.default_rng(SEED)
    original = load_experiment_input()
    modified, injected_labels, used_times = choose_injections(original, rng)
    controls = choose_controls(original, injected_labels, used_times, rng)
    labels = pd.concat([injected_labels, controls], ignore_index=True).sort_values(READING_KEY)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    labels_path = OUTPUT_DIR / "labels.csv"
    detections_path = OUTPUT_DIR / "injected_detections.csv"
    labels.to_csv(labels_path, index=False)
    print(f"Injected {len(injected_labels):,} readings in {len(used_times)} episodes")
    print(f"Selected {len(controls):,} untouched controls")

    torch.set_num_threads(2)
    model, means, stds, sensor_ids, neighbors = load_checkpoint()
    baseline = score_experiment(original, labels, model, means, stds, sensor_ids, neighbors)
    detections = score_experiment(modified, labels, model, means, stds, sensor_ids, neighbors)
    detections.to_csv(detections_path, index=False)
    baseline = prepare_detections(baseline)
    detections = prepare_detections(detections)
    evaluate_labels(detections, labels_path)
    compare_with_baseline(labels, baseline, detections, OUTPUT_DIR)
    print(f"Saved detections, labels, and comparisons to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
