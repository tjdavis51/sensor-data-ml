# preprocess the sensor dataset
# this script, first loads the raw sensor measurements
# then loads sensor positions
# removes records with invalid sensor IDs
# synchronizes measurements into 30 second intervals
# consolidates duplicate measurements
# adds quality flags for suspicious measurements
# attaches spatial coordinates
# saves the processed dataset
# original measurement values are preserved in the original file
# and a new 

import os
import pandas as pd

# path to the original raw dataset
RAW_DATA_PATH = "data/raw/labapp3-data-new"
# path to the original positions dataset
POSITIONS_PATH = "data/raw/positions.txt"
# path to the outputed processed dataset
OUTPUT_PATH = "data/processed/sensor_data.csv"
# bin separation
TIME_INTERVAL = 30
# columns from the oringal dataset
MEASUREMENT_COLUMNS = ["temperature", "humidity", "light", "voltage",]

# function to load in the sensor data to a pandas dataframe
def load_sensor_data(path):
    # columns to load in
    columns = ["time", "nodeid", *MEASUREMENT_COLUMNS,]
    # start the dataframe object with ... and missing values filled with "NaN"
    df = pd.read_csv(path, sep=r"\s+", header=None, names=columns, na_values=["NaN"],)
    # give a return message that it work
    print(f"Loaded {len(df):,} raw entries")
    # give back the dataframe
    return df

# function to load in the positions
def load_sensor_positions(path):
    # read in the position columns with the nodeid
    positions = pd.read_csv(path, sep=r"\s+", header=None, names=["nodeid", "x", "y"],)

    # check if the positions all load in with the corresponding IDs
    if positions["nodeid"].isna().any():
        raise ValueError("Missing sensor IDs in positions file.")
    # and check if the positions generate duplicates
    if positions["nodeid"].duplicated().any():
        raise ValueError("Duplicate sensor IDs in positions file.")

    # if all the positions, print a success message
    print(f"Loaded positions for {len(positions)} sensors.")
    # return the postions
    return positions

# function to filer the sensor IDs, takes in the dataframe and the sensor positions
def filter_sensor_ids(df, positions):
    # save the original count of entries
    original_count = len(df)

    # remove records without valid time or sensor IDs
    df = df.dropna(subset=["time", "nodeid"]).copy()

    # keep only sensors represented in the positions file
    valid_ids = set(positions["nodeid"])

    # update the dataframe to remove sensors without a valid ID
    # identify rows belonging to valid sensors
    valid_mask = df["nodeid"].isin(valid_ids)
    # keep only those rows
    filtered_df = df[valid_mask]
    # create an independent copy
    df = filtered_df.copy()
    # convert IDs to integers
    df["nodeid"] = df["nodeid"].astype(int)

    # get the number of removed rows
    removed = original_count - len(df)

    print(f"Removed {removed:,} records with invalid sensor IDs.")

    return df

# synchronize all the measurements using 30 second time bins
def synchronize_data(df, interval = 30):
    # group observations into fixed-duration time intervals
    # multiple measurements from the same sensor within an interval are averaged

    df = df.copy()

    # assign each timestamp to the beginning of its interval
    df["time"] = (df["time"] // interval) * interval

    # save the original number of entries
    original_count = len(df)

    # group rows that have the same timestamp and sensor ID, each group contains
    # all measurements from one sensor within a certain 30-second interval
    grouped = df.groupby(["time", "nodeid"], as_index=False)
    # select only the measurement columns from each group
    # this stops columns like time and nodeid from averaging
    measurements = grouped[MEASUREMENT_COLUMNS]
    # calculate the mean of each measurement within each group, NaN values are
    # ignored when calculating the mean, if all values for a measurement are NaN,
    # the result remains NaN
    averaged = measurements.mean()
    # replace the original DataFrame with the averaged results
    # there should now be exactly one row per (time, nodeid) combination
    df = averaged

    # get the number of entries that were removed
    consolidated = original_count - len(df)

    print(
        f"Consolidated {consolidated:,} measurements "
        f"into {interval}-second intervals."
    )

    return df

# add in the flags to tell if an entry's measurement is suspicious
def add_quality_flags(df: pd.DataFrame) -> pd.DataFrame:
    
    # flag missing and suspicious measurements
    # suspicious values can still be preserved, but its important to have them marked
    # these ranges are assumptions but a pretty reasonable baseline
    df = df.copy()

    # explicitly track whether each measurement exists
    for column in MEASUREMENT_COLUMNS:
        df[f"{column}_observed"] = df[column].notna()

    # flag any invalid temperature measurements
    # room temperatures should be somewhere in the range 10 - 45 degrees here
    # being very generous with -20 to 60
    df["temperature_suspicious"] = (
        df["temperature"].notna()
        & ~df["temperature"].between(-20, 60)
    )

    # humidity should be between 0 and 100%
    df["humidity_suspicious"] = (
        df["humidity"].notna()
        & ~df["humidity"].between(0, 100)
    )

    # voltage screening for numbers that are not between 1 - 5 volts
    df["voltage_suspicious"] = (
        df["voltage"].notna()
        & ~df["voltage"].between(1, 5)
    )

    print("Suspicious temperature measurements:", df["temperature_suspicious"].sum())

    print("Suspicious humidity measurements:", df["humidity_suspicious"].sum())

    print("Suspicious voltage measurements:", df["voltage_suspicious"].sum())

    return df

# add in the spacial coordinates
def attach_positions(df, positions):

    # many to one, left join of the 2 dataframe on nodeid
    # so keep every row from the left dataframe (which is df) and attach matching
    # information from the right dataframe (which is positions)
    df = df.merge(positions, on="nodeid", how="left", validate="many_to_one")

    # select the x and y coordinate columns
    coordinates = df[["x", "y"]]
    # check each coordinate for missing values
    missing_coordinates = coordinates.isna()
    # check whether either column contains missing values
    missing_by_column = missing_coordinates.any()
    # check whether ay coordinate is missing anywhere
    has_missing_coordinates = missing_by_column.any()
    # stop processing if any measurements lack coordinates
    if has_missing_coordinates:
        raise ValueError(
            "Some measurements have missing sensor coordinates."
        )
    
    # return the measurements with their coordinates attached
    return df

# save the processed data to a csv file
def save_processed_data(df, path):

    # create the output directory if it doesn't already exist
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # save the dataframe as a csv
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

    # sort by time, then sensor ID, and reset the index values
    df = df.sort_values(["time", "nodeid"]).reset_index(drop=True)

    # save the new output
    save_processed_data(df, OUTPUT_PATH)
    # print success
    print("\nPreprocessing complete\n")

if __name__ == "__main__":
    main()