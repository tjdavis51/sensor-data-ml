import os
import json
import numpy as np
import pandas as pd

INPUT_DIR = "data/processed"

OUTPUT_DIR = "data/masked"

MASK_RATE = 0.10

RANDOM_SEED = 42

MEASUREMENT_COLUMNS = ["temperature", "humidity", "light", "voltage",]


def load_dataset(name):

    path = os.path.join(INPUT_DIR, f"{name}.csv")

    df = pd.read_csv(path)

    print(f"Loaded {len(df):,} {name} observations")

    return df


def mask_dataset(df, mask_rate, rng):

    if not 0 <= mask_rate <= 1:
        raise ValueError("Mask rate must be between 0 and 1")

    masked_df = df.copy()

    answers = []

    for column in MEASUREMENT_COLUMNS:

        valid_mask = df[column].notna()

        valid_positions = np.flatnonzero(valid_mask.to_numpy())

        valid_count = len(valid_positions)

        mask_count = int(valid_count * mask_rate)

        selected_positions = rng.choice(
            valid_positions,
            size=mask_count,
            replace=False
        )

        original_values = df[column].iloc[selected_positions].to_numpy()

        times = df["time"].iloc[selected_positions].to_numpy()

        nodeids = df["nodeid"].iloc[selected_positions].to_numpy()

        for time, nodeid, value in zip(times, nodeids, original_values):

            answers.append({
                "time": time,
                "nodeid": nodeid,
                "feature": column,
                "true_value": value,
            })

        column_index = masked_df.columns.get_loc(column)
        masked_df.iloc[selected_positions, column_index] = np.nan

        print(f"  {column}: {mask_count:,} / {valid_count:,} masked")

    answers_df = pd.DataFrame(
        answers,
        columns=["time", "nodeid", "feature", "true_value"]
    )

    answers_df = answers_df.sort_values(
        ["time", "nodeid", "feature"]
    )

    answers_df = answers_df.reset_index(drop=True)

    answers_df.insert(0, "target_id", np.arange(len(answers_df)))

    return masked_df, answers_df


def save_dataset(name, masked_df, answers_df):

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    masked_path = os.path.join(OUTPUT_DIR, f"{name}_masked.csv")
    answers_path = os.path.join(OUTPUT_DIR, f"{name}_answers.csv")

    masked_df.to_csv(masked_path, index=False)

    answers_df.to_csv(answers_path, index=False)

    print(f"Saved {name}_masked.csv and {name}_answers.csv")


def save_config():

    config = {
        "mask_rate": MASK_RATE,
        "random_seed": RANDOM_SEED,
        "masking_strategy": "random",
        "measurement_columns": MEASUREMENT_COLUMNS,
    }

    path = os.path.join(OUTPUT_DIR, "masking_config.json")

    with open(path, "w") as file:
        json.dump(config, file, indent=4)

    print("Saved masking_config.json")


def main():

    print("\nMasking Sensor Datasets:\n")

    print(f"Mask rate: {MASK_RATE * 100:.2f}%")
    print(f"Random seed: {RANDOM_SEED}")

    rng = np.random.default_rng(RANDOM_SEED)

    for name in ["train", "val", "test"]:

        print(f"\n{name.upper()} dataset:")

        df = load_dataset(name)

        masked_df, answers_df = mask_dataset(df, MASK_RATE, rng)

        save_dataset(name, masked_df, answers_df)

    save_config()

    print("\nDataset masking complete")


if __name__ == "__main__":
    main()
