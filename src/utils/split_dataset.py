import os
import numpy as np
import pandas as pd

INPUT_PATH = "data/processed/sensor_data.csv"
OUTPUT_DIR = "data/processed"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

NORMAL_DAYS = 20
SECONDS_PER_DAY = 24 * 60 * 60

MEASUREMENT_COLUMNS = ["temperature", "humidity", "light", "voltage",]

def load_dataset(path):

    df = pd.read_csv(path)

    print(f"Loaded {len(df):,} processed observations")

    return df

def mask_suspicious_values(df):

    df = df.copy()

    for column in MEASUREMENT_COLUMNS:

        flag_column = f"{column}_suspicious"

        if flag_column not in df.columns:
            continue

        suspicious = df[flag_column].fillna(False).astype(bool)

        count = suspicious.sum()

        df.loc[suspicious, column] = np.nan

        print(f"Masked {count:,} suspicious {column} measurements")

    return df


def chronological_split(df, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO):

    if not 0 < train_ratio < 1:
        raise ValueError("Invalid training ratio")

    if not 0 < val_ratio < 1:
        raise ValueError("Invalid validation ratio")

    if train_ratio + val_ratio >= 1:
        raise ValueError("Training and validation ratios must sum to less than 1")

    timestamps = np.sort(df["time"].unique())

    n_times = len(timestamps)

    train_end = int(n_times * train_ratio)
    val_end = int(n_times * (train_ratio + val_ratio))
    train_boundary = timestamps[train_end]
    val_boundary = timestamps[val_end]
    train_mask = df["time"] < train_boundary
    train_df = df[train_mask].copy()

    val_start_mask = df["time"] >= train_boundary
    val_end_mask = df["time"] < val_boundary
    val_mask = val_start_mask & val_end_mask
    val_df = df[val_mask].copy()

    test_mask = df["time"] >= val_boundary
    test_df = df[test_mask].copy()

    return train_df, val_df, test_df


def print_split_summary(name, df):

    print(f"\n{name.upper()} dataset:")

    print(f"Rows: {len(df):,}")

    print(f"Unique timestamps: {df['time'].nunique():,}")

    print(f"Sensors: {df['nodeid'].nunique()}")

    if len(df) > 0:

        start_time = df["time"].min()

        end_time = df["time"].max()

        print(f"Time range: {start_time} to {end_time}")

    print("\nAvailable measurements:")

    for column in MEASUREMENT_COLUMNS:

        valid_count = df[column].notna().sum()

        total_count = len(df)

        percentage = 100 * valid_count / total_count if total_count else 0

        print(
            f"  {column}: "
            f"{valid_count:,} / {total_count:,} "
            f"({percentage:.2f}%)"
        )


def save_splits(train_df, val_df, test_df):

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for name, frame in (("train", train_df), ("val", val_df), ("test", test_df)):
        frame.to_csv(os.path.join(OUTPUT_DIR, f"{name}.csv"), index=False)

    print("\nSaved train.csv, val.csv, and test.csv")


def main():

    print("\nSplitting Sensor Dataset:\n")

    df = load_dataset(INPUT_PATH)

    normal_end = NORMAL_DAYS * SECONDS_PER_DAY

    normal_mask = df["time"] < normal_end

    normal_df = df[normal_mask].copy()

    anomaly_df = df[~normal_mask].copy()

    print(f"Normal observations: {len(normal_df):,}")
    print(f"Anomaly period observations: {len(anomaly_df):,}")

    normal_df = mask_suspicious_values(normal_df)

    train_df, val_df, test_df = chronological_split(normal_df)

    print_split_summary("train", train_df)
    print_split_summary("validation", val_df)
    print_split_summary("test", test_df)

    save_splits(train_df, val_df, test_df)

    anomaly_path = os.path.join(OUTPUT_DIR, "anomaly_period.csv")

    anomaly_df.to_csv(anomaly_path, index=False)

    print("\nSaved anomaly_period.csv")

    print("\nDataset splitting complete")


if __name__ == "__main__":
    main()
