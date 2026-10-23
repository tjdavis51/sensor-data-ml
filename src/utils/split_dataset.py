# split the processed sensor dataset into training, validation, and testing datasets
# this script first loads the processed sensor measurements
# separates the normal operating period from the later observations
# replaces suspicious measurement values with NaN in the normal dataset
# splits the normal dataset chronologically using unique timestamps
# prints statistics for each dataset
# saves the three datasets as separate csv files
# saves the remaining observations for anomaly evaluation
# all models will use the same datasets and cleaning rules

import os
import numpy as np
import pandas as pd

# path to the processed sensor dataset
INPUT_PATH = "data/processed/sensor_data.csv"
# directory where the split datasets will be saved
OUTPUT_DIR = "data/processed"

# percentages of the dataset to use for training, validation, and testing
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# number of days to use for normal reconstruction experiments
NORMAL_DAYS = 20
# number of seconds in one day
SECONDS_PER_DAY = 24 * 60 * 60

# columns containing the sensor measurements
MEASUREMENT_COLUMNS = ["temperature", "humidity", "light", "voltage",]

# function to load the processed sensor dataset
def load_dataset(path):

    # read the processed csv file into a pandas dataframe
    df = pd.read_csv(path)

    # print the number of observations loaded
    print(f"Loaded {len(df):,} processed observations")

    # return the dataframe
    return df

# function to replace suspicious measurements with NaN
def mask_suspicious_values(df):

    # create an independent copy of the dataframe
    df = df.copy()

    # go through each measurement column
    for column in MEASUREMENT_COLUMNS:

        # get the name of the corresponding suspicious flag column
        flag_column = f"{column}_suspicious"

        # skip this measurement if it does not have a suspicious flag column
        if flag_column not in df.columns:
            continue

        # get the suspicious flags for this measurement
        # replace missing flags with False and convert the values to booleans
        suspicious = df[flag_column].fillna(False).astype(bool)

        # count the number of suspicious measurements
        count = suspicious.sum()

        # replace only the suspicious measurements with NaN
        # other measurements from the same observation can be kept
        df.loc[suspicious, column] = np.nan

        # print the number of measurements that were masked
        print(f"Masked {count:,} suspicious {column} measurements")

    # return the dataframe with suspicious values replaced
    return df


# function to split the dataset chronologically
def chronological_split(df, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO):

    # check that the training ratio is between 0 and 1
    if not 0 < train_ratio < 1:
        raise ValueError("Invalid training ratio")

    # check that the validation ratio is between 0 and 1
    if not 0 < val_ratio < 1:
        raise ValueError("Invalid validation ratio")

    # check that there is still data remaining for the testing dataset
    if train_ratio + val_ratio >= 1:
        raise ValueError("Training and validation ratios must sum to less than 1")

    # get all unique timestamps and sort them in chronological order
    timestamps = np.sort(df["time"].unique())

    # get the total number of unique timestamps
    n_times = len(timestamps)

    # calculate the index where the training dataset ends
    train_end = int(n_times * train_ratio)
    # calculate the index where the validation dataset ends
    val_end = int(n_times * (train_ratio + val_ratio))
    # get the timestamp where the validation dataset begins
    train_boundary = timestamps[train_end]
    # get the timestamp where the testing dataset begins
    val_boundary = timestamps[val_end]
    # identify rows with timestamps before the validation dataset
    train_mask = df["time"] < train_boundary
    # keep only those rows and create an independent copy
    train_df = df[train_mask].copy()

    # identify rows with timestamps after the training dataset
    val_start_mask = df["time"] >= train_boundary
    # identify rows with timestamps before the testing dataset
    val_end_mask = df["time"] < val_boundary
    # combine the two conditions to identify validation rows
    val_mask = val_start_mask & val_end_mask
    # keep only the validation rows and create an independent copy
    val_df = df[val_mask].copy()

    # identify rows with timestamps after the validation dataset
    test_mask = df["time"] >= val_boundary
    # keep only the testing rows and create an independent copy
    test_df = df[test_mask].copy()

    # return all three datasets
    return train_df, val_df, test_df


# function to print statistics about each dataset
def print_split_summary(name, df):

    # print the name of the dataset
    print(f"\n{name.upper()} dataset:")

    # print the number of rows
    print(f"Rows: {len(df):,}")

    # print the number of unique timestamps
    print(f"Unique timestamps: {df['time'].nunique():,}")

    # print the number of unique sensors
    print(f"Sensors: {df['nodeid'].nunique()}")

    # only print the time range if the dataset contains observations
    if len(df) > 0:

        # get the earliest timestamp
        start_time = df["time"].min()

        # get the latest timestamp
        end_time = df["time"].max()

        # print the time range
        print(f"Time range: {start_time} to {end_time}")

    # print a heading for the measurement statistics
    print("\nAvailable measurements:")

    # go through each measurement column
    for column in MEASUREMENT_COLUMNS:

        # count the number of measurements that are not NaN
        valid_count = df[column].notna().sum()

        # count the total number of observations
        total_count = len(df)

        # calculate the percentage of available measurements
        # use 0 if the dataset is empty to avoid division by zero
        percentage = 100 * valid_count / total_count if total_count else 0

        # print the number and percentage of available measurements
        print(
            f"  {column}: "
            f"{valid_count:,} / {total_count:,} "
            f"({percentage:.2f}%)"
        )


# function to save the training, validation, and testing datasets
def save_splits(train_df, val_df, test_df):

    # create the output directory if it does not already exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # create the output paths for each dataset
    train_path = os.path.join(OUTPUT_DIR, "train.csv")
    val_path = os.path.join(OUTPUT_DIR, "val.csv")
    test_path = os.path.join(OUTPUT_DIR, "test.csv")

    # save the training dataset
    train_df.to_csv(train_path, index=False)

    # save the validation dataset
    val_df.to_csv(val_path, index=False)

    # save the testing dataset
    test_df.to_csv(test_path, index=False)

    # print a success message
    print("\nSaved train.csv, val.csv, and test.csv")


# main function to run the dataset splitting process
def main():

    print("\nSplitting Sensor Dataset:\n")

    # load the processed sensor dataset
    df = load_dataset(INPUT_PATH)

    # calculate the end of the normal operating period
    normal_end = NORMAL_DAYS * SECONDS_PER_DAY

    # identify observations from the first specified number of days
    normal_mask = df["time"] < normal_end

    # keep the normal operating period for reconstruction
    normal_df = df[normal_mask].copy()

    # keep the remaining observations for anomaly evaluation
    anomaly_df = df[~normal_mask].copy()

    # print the number of observations in each period
    print(f"Normal observations: {len(normal_df):,}")
    print(f"Anomaly period observations: {len(anomaly_df):,}")

    # replace suspicious measurements with NaN in the normal dataset
    normal_df = mask_suspicious_values(normal_df)

    # split the normal dataset chronologically
    train_df, val_df, test_df = chronological_split(normal_df)

    # print statistics for each dataset
    print_split_summary("train", train_df)
    print_split_summary("validation", val_df)
    print_split_summary("test", test_df)

    # save the three datasets to csv files
    save_splits(train_df, val_df, test_df)

    # create the output path for the anomaly period dataset
    anomaly_path = os.path.join(OUTPUT_DIR, "anomaly_period.csv")

    # save the remaining observations without masking suspicious values
    anomaly_df.to_csv(anomaly_path, index=False)

    # print a success message
    print("\nSaved anomaly_period.csv")

    # print a success message
    print("\nDataset splitting complete")


# run main() only when this file is executed directly
if __name__ == "__main__":
    main()