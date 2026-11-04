# generate reproducible masks and hide the target measurements

# to keep things fair and for a apples to apples comparison
# each implementation needs to follow:
# 1. same training, validation, and testing periods
# 2. same hidden measurements during validation and testing
# 3. same information available when masking predictions
# 4. same evaluation metrics applied to the same predictions

# randomly mask measurements in the training, validation, and testing datasets
# this script loads the three processed datasets
# selects a configurable percentage of valid measurements to hide
# replaces the selected measurements with NaN
# saves the original values in separate answer files
# saves the masked datasets and answers to data/masked
# uses a fixed random seed so the masking is reproducible

import os
import json
import numpy as np
import pandas as pd

# directory containing the processed datasets
INPUT_DIR = "data/processed"

# directory where the masked datasets will be saved
OUTPUT_DIR = "data/masked"

# percentage of valid measurements to mask
MASK_RATE = 0.10

# random seed for reproducible masking
RANDOM_SEED = 42

# columns containing the sensor measurements
MEASUREMENT_COLUMNS = ["temperature", "humidity", "light", "voltage",]


# function to load a processed dataset
def load_dataset(name):

    # create the path to the dataset
    path = os.path.join(INPUT_DIR, f"{name}.csv")

    # read the csv file into a pandas dataframe
    df = pd.read_csv(path)

    # print the number of observations loaded
    print(f"Loaded {len(df):,} {name} observations")

    # return the dataframe
    return df


# function to randomly mask valid measurements
def mask_dataset(df, mask_rate, rng):

    # check that the masking rate is between 0 and 1
    if not 0 <= mask_rate <= 1:
        raise ValueError("Mask rate must be between 0 and 1")

    # create an independent copy of the dataset
    masked_df = df.copy()

    # create a list to store the hidden measurements
    answers = []

    # go through each measurement column
    for column in MEASUREMENT_COLUMNS:

        # identify measurements that are not NaN
        valid_mask = df[column].notna()

        # get the row positions of valid measurements
        valid_positions = np.flatnonzero(valid_mask.to_numpy())

        # count the number of valid measurements
        valid_count = len(valid_positions)

        # calculate how many measurements to hide
        mask_count = int(valid_count * mask_rate)

        # randomly select measurements without replacement
        selected_positions = rng.choice(
            valid_positions,
            size=mask_count,
            replace=False
        )

        # get the original values of the selected measurements
        original_values = df[column].iloc[selected_positions].to_numpy()

        # get the timestamps of the selected measurements
        times = df["time"].iloc[selected_positions].to_numpy()

        # get the sensor IDs of the selected measurements
        nodeids = df["nodeid"].iloc[selected_positions].to_numpy()

        # go through each selected measurement
        for time, nodeid, value in zip(times, nodeids, original_values):

            # save the location and original value of the measurement
            answers.append({
                "time": time,
                "nodeid": nodeid,
                "feature": column,
                "true_value": value,
            })

        # replace the selected measurements with NaN
        column_index = masked_df.columns.get_loc(column)
        masked_df.iloc[selected_positions, column_index] = np.nan

        # print the number of measurements masked
        print(f"  {column}: {mask_count:,} / {valid_count:,} masked")

    # create a dataframe containing the original measurements
    answers_df = pd.DataFrame(
        answers,
        columns=["time", "nodeid", "feature", "true_value"]
    )

    # sort the answers by timestamp, sensor ID, and feature
    answers_df = answers_df.sort_values(
        ["time", "nodeid", "feature"]
    )

    # reset the row indices
    answers_df = answers_df.reset_index(drop=True)

    # assign a unique ID to each prediction target
    answers_df.insert(0, "target_id", np.arange(len(answers_df)))

    # return the masked dataset and answer key
    return masked_df, answers_df


# function to save the masked dataset and answer key
def save_dataset(name, masked_df, answers_df):

    # create the output directory if it does not already exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # create the paths for the output files
    masked_path = os.path.join(OUTPUT_DIR, f"{name}_masked.csv")
    answers_path = os.path.join(OUTPUT_DIR, f"{name}_answers.csv")

    # save the masked dataset
    masked_df.to_csv(masked_path, index=False)

    # save the answer key
    answers_df.to_csv(answers_path, index=False)

    # print a success message
    print(f"Saved {name}_masked.csv and {name}_answers.csv")


# function to save the masking configuration
def save_config():

    # create the masking configuration
    config = {
        "mask_rate": MASK_RATE,
        "random_seed": RANDOM_SEED,
        "masking_strategy": "random",
        "measurement_columns": MEASUREMENT_COLUMNS,
    }

    # create the path for the configuration file
    path = os.path.join(OUTPUT_DIR, "masking_config.json")

    # save the configuration as a json file
    with open(path, "w") as file:
        json.dump(config, file, indent=4)

    # print a success message
    print("Saved masking_config.json")


# main function to mask all three datasets
def main():

    print("\nMasking Sensor Datasets:\n")

    # print the masking configuration
    print(f"Mask rate: {MASK_RATE * 100:.2f}%")
    print(f"Random seed: {RANDOM_SEED}")

    # create the random number generator
    rng = np.random.default_rng(RANDOM_SEED)

    # go through the training, validation, and testing datasets
    for name in ["train", "val", "test"]:

        # print the name of the dataset
        print(f"\n{name.upper()} dataset:")

        # load the processed dataset
        df = load_dataset(name)

        # randomly mask valid measurements
        masked_df, answers_df = mask_dataset(df, MASK_RATE, rng)

        # save the masked dataset and answer key
        save_dataset(name, masked_df, answers_df)

    # save the masking configuration
    save_config()

    # print a success message
    print("\nDataset masking complete")


# run main() only when this file is executed directly
if __name__ == "__main__":
    main()