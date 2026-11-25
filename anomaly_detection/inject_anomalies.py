# controlled anomaly injection for the gru detector
#
# input from held-out sensor readings and their original surrounding context
# seeded selection of invalid humidity light spikes temperature offsets and stuck humidity
# untouched controls outside the windows affected by each injection
# original and modified scoring with the same saved gru and frozen feature thresholds
# output csv files with labels detections and comparisons at reading and episode level
# run settings at the top of this file for a no-argument experiment

# load the helpers for locating the repo and importing the other scripts
from pathlib import Path
import sys

# load the libraries for sampling csv files and gru inference
import numpy as np
import pandas as pd
import torch

# find the repo root from the location of this file
ROOT = Path(__file__).resolve().parents[1]
# add the repo root to the import path only for direct script runs
if __package__ in (None, ""):
    # let python find the detector when this script is run by file path
    sys.path.insert(0, str(ROOT))

# reuse the model setup rules and scoring from the detector
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
# reuse the evaluator's reading keys and flag definitions
from anomaly_detection.evaluate_anomaly_detection import (
    DETECTORS,
    READING_KEY,
    evaluate_labels,
    prepare_detections,
)


# choose a scoring interval inside the reconstruction test split
# use elapsed seconds with an inclusive start and exclusive end
START_TIME = 1423560
# stop at the end of the held-out test period by default
END_TIME = 1728000
# save the controlled experiment in its own results folder
OUTPUT_DIR = ROOT / "anomaly_detection/results/injection"
# use a fixed seed for repeatable injections and controls
SEED = 42
# request the same number of episodes for each injection type
EPISODES_PER_TYPE = 8
# sample controls in proportion to the changed readings for each feature
CONTROLS_PER_INJECTED_READING = 2

# use native units and keep the contextual cases within the detector's rule bounds
# stuck humidity holds its first value for the rest of the episode
INJECTIONS = [
    ("stuck_humidity", "humidity", 10, None),
    ("temperature_offset", "temperature", 10, 4.0),
    ("light_spike", "light", 1, 400.0),
    ("invalid_humidity", "humidity", 1, -5.0),
]
# require enough original movement before freezing a humidity sequence
MIN_STUCK_CHANGE = 0.3
# cap injected light to keep the spike within the chosen experiment range
MAX_INJECTED_LIGHT = 1800.0


# load the original sensor observations for the experiment
def load_experiment_input():
    # limit the experiment to the test period and keep context around the scoring slice
    test = pd.read_csv(PROCESSED_DIR / "test.csv", usecols=["time", "nodeid"])
    # find the first available test timestamp
    test_start = test["time"].min()
    # make the test end exclusive by adding one timestep
    test_end = test["time"].max() + TIME_STEP
    # keep the requested experiment interval inside the held-out data
    if not test_start <= START_TIME < END_TIME <= test_end:
        raise ValueError("The injection interval must be within the test split")

    # set aside enough surrounding time to build full model windows
    padding = WINDOW_LENGTH * TIME_STEP
    # limit the preceding context to the test split
    context_start = max(test_start, START_TIME - padding)
    # limit the following context to the test split
    context_end = min(test_end, END_TIME + padding)
    # recover the aggregates before reconstruction-specific cleaning
    source = pd.read_csv(PROCESSED_DIR / "sensor_data.csv", usecols=INPUT_COLUMNS)
    # keep the requested interval and its allowed context
    context = source[source["time"].ge(context_start) & source["time"].lt(context_end)]
    # sort the readings so each sensor's sequence follows time order
    return context.sort_values(["time", "nodeid"]).reset_index(drop=True)


# choose where to inject each type of synthetic anomaly
def choose_injections(original, rng):
    # apply both rule layers before choosing original readings
    invalid, heuristic = quality_rules(original)
    # copy the input so baseline readings remain available
    modified = original.copy()
    # restrict changes to the scoring interval
    within_slice = original["time"].ge(START_TIME) & original["time"].lt(END_TIME)
    # track occupied episode times to prevent shared model context
    used_times = []
    # record a label for each changed measurement
    labels = []

    # work through the configured injection types
    for anomaly_type, feature, length, amount in INJECTIONS:
        # start with observed readings that pass both rule layers
        usable = original[within_slice & original[feature].notna()
                          & invalid[feature].eq("") & heuristic[feature].eq("")]
        # collect valid candidate windows before sampling
        candidates = []
        # search within each sensor so a window stays on the same sensor
        for _, sensor in usable.groupby("nodeid"):
            # take the elapsed timestamps for the sensor
            times = sensor["time"].to_numpy()
            # take the feature readings to modify
            values = sensor[feature].to_numpy()
            # keep the source row indices for the eventual edit
            indices = sensor.index.to_numpy()
            # check every possible starting position for a full episode
            for start in range(len(sensor) - length + 1):
                # calculate the exclusive end position of this candidate
                stop = start + length
                # skip any candidate window with missing timesteps
                if times[stop - 1] - times[start] != (length - 1) * TIME_STEP:
                    continue

                # use the first reading as the anchor for the stuck episode
                current = values[start:stop]
                # handle the stuck case by repeating its starting value
                if anomaly_type == "stuck_humidity":
                    # skip a candidate that was already nearly constant
                    if np.max(np.abs(current - current[0])) < MIN_STUCK_CHANGE:
                        continue
                    # use the first value throughout the candidate window
                    injected = np.full(length, current[0])
                # handle physical invalidity as a replacement value
                elif anomaly_type == "invalid_humidity":
                    # fill the candidate with the configured invalid humidity
                    injected = np.full(length, amount)
                else:
                    # add the spike or offset to the current readings
                    injected = current + amount
                    # keep the shifted temperatures inside the upper heuristic bound
                    if feature == "temperature" and (injected > 60).any():
                        continue
                    # keep light spikes below the configured upper limit
                    if feature == "light" and (injected > MAX_INJECTED_LIGHT).any():
                        continue

                # retain the source rows replacement values and time bounds
                candidates.append((indices[start:stop], injected, times[start], times[stop - 1]))

        # start counting the accepted episodes for this type
        count = 0
        # visit candidates in reproducible random order
        for candidate in rng.permutation(len(candidates)):
            # recover the window details for this candidate
            indices, injected, first_time, last_time = candidates[candidate]

            # keep episodes far enough apart that they cannot share a gru window
            padding = WINDOW_LENGTH * TIME_STEP
            # compare the candidate with all episodes accepted so far
            if any(first_time <= end + padding and last_time >= start - padding
                   for start, end in used_times):
                continue

            # count this episode and assign it a stable id
            count += 1
            episode_id = f"{anomaly_type}_{count:02d}"
            # reserve the whole episode span including any unchanged anchor
            used_times.append((first_time, last_time))
            # apply each changed value and record its label
            for index, value in zip(indices, injected):
                # keep the original value for the paired comparison
                before = original.at[index, feature]
                # omit unchanged cells such as the anchor in a stuck episode
                if value == before:
                    continue

                # write the changed measurement into the experiment copy
                modified.at[index, feature] = value
                # store the reading key and exact values before and after the edit
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

            # move to the next type after placing the requested episodes
            if count == EPISODES_PER_TYPE:
                break

        # stop if the interval could not support the requested placement
        if count != EPISODES_PER_TYPE:
            raise ValueError(
                f"Placed {count} of {EPISODES_PER_TYPE} {anomaly_type} episodes "
                "with the required separation. Increase the interval or lower the episode count."
            )

    # return the experiment data and the information used to choose controls
    return modified, pd.DataFrame(labels), used_times


# sample untouched readings for the labeled comparison
def choose_controls(original, injected_labels, used_times, rng):
    # apply the same rule eligibility used for injection candidates
    invalid, heuristic = quality_rules(original)
    # start with timestamps within the scoring interval
    eligible_times = original["time"].ge(START_TIME) & original["time"].lt(END_TIME)

    # keep controls outside windows that could contain any injected reading
    padding = WINDOW_LENGTH * TIME_STEP
    # remove the time around each injection from the control pool
    for start, end in used_times:
        eligible_times &= ~original["time"].between(start - padding, end + padding)

    # collect the control labels separately
    controls = []
    # choose controls in proportion to the injected readings of each feature
    for feature, count in injected_labels.groupby("feature").size().items():
        # require an observed value with no original rule alert
        eligible = original[eligible_times & original[feature].notna()
                            & invalid[feature].eq("") & heuristic[feature].eq("")]
        # calculate how many controls are needed for this feature
        sample_size = count * CONTROLS_PER_INJECTED_READING
        # fail when the requested control count exceeds the eligible readings
        if len(eligible) < sample_size:
            raise ValueError(f"Need {sample_size} eligible control readings for {feature}")

        # sample the controls without replacement
        positions = rng.choice(len(eligible), size=sample_size, replace=False)
        # turn each selected source row into a control label
        for row in eligible.iloc[positions].itertuples(index=False):
            # get the observed value of the requested feature
            value = getattr(row, feature)
            # store the same value before and after because the reading stays untouched
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

    # return the labels for the selected controls
    return pd.DataFrame(controls)


# score the labeled readings from one version of the input
def score_experiment(observations, labels, model, means, stds, sensor_ids, neighbors):
    # keep the full window context and only predict the readings used in the comparison
    invalid, heuristic = quality_rules(observations)
    # build the normalized context using the same physical screening as detection
    values, availability, times = prepare_inputs(observations, invalid, means, stds, sensor_ids)
    # reshape the measurements and their reasons into individual reading rows
    readings = reading_rows(observations, invalid, heuristic)
    # select the labeled readings using the full reading key
    targets = labels[READING_KEY].merge(readings, on=READING_KEY, validate="one_to_one")
    # confirm that every requested reading had a matching source measurement
    if len(targets) != len(labels):
        raise ValueError("Some experiment targets are absent from the input")

    # apply the existing residual scores and combined flag
    return score_readings(
        targets, model, values, availability, times, sensor_ids, neighbors, means, stds
    )


# compare alerts before and after injection at reading and episode level
def compare_with_baseline(labels, baseline, detections, output_dir):
    # take the flag columns for rules residuals and the combined detector
    flag_columns = list(DETECTORS.values())
    # join labels to both sets of results on the reading key
    comparison = labels.merge(
        baseline[READING_KEY + flag_columns], on=READING_KEY, validate="one_to_one"
    ).merge(
        detections[READING_KEY + flag_columns], on=READING_KEY, validate="one_to_one",
        suffixes=("_baseline", "_injected")
    )
    # verify that both sets of detections covered every label
    if len(comparison) != len(labels):
        raise ValueError("Some labels are missing from the original or modified detections")

    # show alerts that appear after injection and keep any alerts that were already present
    positives = comparison[comparison["is_injected"]].copy()
    # group changed readings by injection type and episode
    groups = positives.groupby(["anomaly_type", "episode_id"])
    # record the affected times sensor feature and number of changed readings
    episodes = groups.agg(
        time_start=("time", "min"), time_end=("time", "max"),
        nodeid=("nodeid", "first"), feature=("feature", "first"),
        changed_readings=("time", "size"),
    )
    # start the episode summary with the number injected per type
    summary = episodes.groupby(level="anomaly_type").size().rename("episodes").to_frame()
    # fill in the episode counts for each detector
    for name, flag in DETECTORS.items():
        # identify the alert column from the original readings
        before = f"{flag}_baseline"
        # identify the alert column from the modified readings
        after = f"{flag}_injected"
        # name the column used for alerts first seen after injection
        new = f"{name}_new_alert"
        # mark a reading as newly flagged only when its baseline flag was false
        comparison[new] = comparison[after] & ~comparison[before]
        # mark an episode already flagged if any of its labeled readings fired before injection
        episodes[f"{name}_baseline"] = groups[before].any()
        # mark an episode detected if any changed reading fired after injection
        episodes[f"{name}_detected"] = groups[after].any()
        # count an episode as newly detected only when it had no baseline alert
        episodes[new] = episodes[f"{name}_detected"] & ~episodes[f"{name}_baseline"]
        # count detected episodes for each injection type
        summary[f"{name}_detected"] = episodes.groupby(level="anomaly_type")[f"{name}_detected"].sum()
        # count newly detected episodes for each injection type
        summary[f"{name}_new"] = episodes.groupby(level="anomaly_type")[new].sum()

    # save the paired flags for each injected reading and control
    comparison.to_csv(output_dir / "reading_comparison.csv", index=False)
    # save the event-level flags and the bounds of the changed readings
    episodes.to_csv(output_dir / "episode_comparison.csv")
    # print the label for the episode summary
    print("\nEpisode comparison")
    # show the episode counts by detection layer
    print(summary.to_string())
    # explain how the summary defines a new episode alert
    print("A new episode alert has a flag after injection and no baseline flag on its labeled readings")


# set up the experiment and run both detection passes
def main():
    # start the shared seeded random sampler
    rng = np.random.default_rng(SEED)
    # load the original held-out data
    original = load_experiment_input()
    # make the modified copy and keep track of the affected times
    modified, injected_labels, used_times = choose_injections(original, rng)
    # choose control readings outside the affected windows
    controls = choose_controls(original, injected_labels, used_times, rng)
    # combine the positive and control labels in key order
    labels = pd.concat([injected_labels, controls], ignore_index=True).sort_values(READING_KEY)

    # create the experiment output folder
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # choose the file for the reading-level labels
    labels_path = OUTPUT_DIR / "labels.csv"
    # choose the file for the modified detection results
    detections_path = OUTPUT_DIR / "injected_detections.csv"
    # save the exact reading keys for the controlled evaluation
    labels.to_csv(labels_path, index=False)
    # report the number of changed readings and episodes
    print(f"Injected {len(injected_labels):,} readings in {len(used_times)} episodes")
    # report the number of untouched controls
    print(f"Selected {len(controls):,} untouched controls")

    # reuse the detector and its frozen thresholds for both passes
    torch.set_num_threads(2)
    # load one checkpoint for both versions of the experiment
    model, means, stds, sensor_ids, neighbors = load_checkpoint()
    # score the labeled readings with their original observations
    baseline = score_experiment(original, labels, model, means, stds, sensor_ids, neighbors)
    # score the same keys using the modified context
    detections = score_experiment(modified, labels, model, means, stds, sensor_ids, neighbors)
    # save the modified detections for standalone evaluation
    detections.to_csv(detections_path, index=False)
    # prepare both sets of detections for the paired comparison
    baseline = prepare_detections(baseline)
    detections = prepare_detections(detections)
    # run the existing reading-level evaluation against the experiment labels
    evaluate_labels(detections, labels_path)
    # compare both sets of flags and summarize the episodes
    compare_with_baseline(labels, baseline, detections, OUTPUT_DIR)
    # point to the folder containing the saved experiment outputs
    print(f"Saved detections, labels, and comparisons to {OUTPUT_DIR}")


# run the entry point when this file is started directly
if __name__ == "__main__":
    main()
