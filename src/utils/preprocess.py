import os
import pandas as pd

RAW_DATA_PATH = "data/raw/labapp3-data-new"
POSITIONS_PATH = "data/raw/positions.txt"
OUTPUT_PATH = "data/processed/sensor_data.csv"
TIME_INTERVAL = 30
MEASUREMENT_COLUMNS = ["temperature", "humidity", "light", "voltage",]

def load_sensor_data(path):
    columns = ["time", "nodeid", *MEASUREMENT_COLUMNS,]
    df = pd.read_csv(path, sep=r"\s+", header=None, names=columns, na_values=["NaN"],)
    print(f"Loaded {len(df):,} raw entries")
    return df

def load_sensor_positions(path):
    positions = pd.read_csv(path, sep=r"\s+", header=None, names=["nodeid", "x", "y"],)

    if positions["nodeid"].isna().any():
        raise ValueError("Missing sensor IDs in positions file.")
    if positions["nodeid"].duplicated().any():
        raise ValueError("Duplicate sensor IDs in positions file.")

    print(f"Loaded positions for {len(positions)} sensors.")
    return positions

def filter_sensor_ids(df, positions):
    original_count = len(df)

    df = df.dropna(subset=["time", "nodeid"]).copy()

    df = df.loc[df["nodeid"].isin(positions["nodeid"])].copy()
    df["nodeid"] = df["nodeid"].astype(int)

    removed = original_count - len(df)

    print(f"Removed {removed:,} records with invalid sensor IDs.")

    return df

def synchronize_data(df, interval = 30):

    df = df.copy()

    df["time"] = (df["time"] // interval) * interval

    original_count = len(df)

    grouped = df.groupby(["time", "nodeid"], as_index=False)
    df = grouped[MEASUREMENT_COLUMNS].mean()

    consolidated = original_count - len(df)

    print(
        f"Consolidated {consolidated:,} measurements "
        f"into {interval}-second intervals."
    )

    return df

def add_quality_flags(df: pd.DataFrame) -> pd.DataFrame:
    
    df = df.copy()

    for column in MEASUREMENT_COLUMNS:
        df[f"{column}_observed"] = df[column].notna()

    limits = {"temperature": (-20, 60), "humidity": (0, 100), "voltage": (1, 5)}
    for column, (low, high) in limits.items():
        values = df[column]
        flag = f"{column}_suspicious"
        df[flag] = values.notna() & ~values.between(low, high)
        print(f"Suspicious {column} measurements:", df[flag].sum())

    return df

def attach_positions(df, positions):

    df = df.merge(positions, on="nodeid", how="left", validate="many_to_one")

    if df[["x", "y"]].isna().any().any():
        raise ValueError(
            "Some measurements have missing sensor coordinates."
        )
    
    return df

def save_processed_data(df, path):

    os.makedirs(os.path.dirname(path), exist_ok=True)

    df.to_csv(path, index=False)

    print(f"Saved {len(df):,} processed observations to {path}")

def main():
    print("\nPreprocessing Sensor Data:\n")

    df = load_sensor_data(RAW_DATA_PATH)
    positions = load_sensor_positions(POSITIONS_PATH)
    df = filter_sensor_ids(df, positions)
    df = synchronize_data(df, TIME_INTERVAL)
    df = add_quality_flags(df)
    df = attach_positions(df, positions)

    df = df.sort_values(["time", "nodeid"]).reset_index(drop=True)

    save_processed_data(df, OUTPUT_PATH)
    print("\nPreprocessing complete\n")

if __name__ == "__main__":
    main()
