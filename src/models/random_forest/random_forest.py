"""Random Forest baseline for predicting masked sensor readings."""

from pathlib import Path
import argparse
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


ROOT = Path(__file__).resolve().parents[3]
MASKED_DIR = ROOT / "data/masked"
RESULTS_DIR = ROOT / "src/models/random_forest/results"
FEATURES = ["temperature", "humidity", "light", "voltage"]
RANDOM_STATE = 42


def make_features(data, target):
    """Build predictors without including the measurement being predicted."""
    # Other measurements and sensor location may help explain the missing value.
    other = [feature for feature in FEATURES if feature != target]
    columns = ["time", "nodeid", "x", "y"] + other
    inputs = data[columns].copy()

    # Use readings from this same sensor at exactly +/-30 seconds.
    # Missing or masked neighbors stay NaN rather than using a distant reading.
    readings = data.set_index(["nodeid", "time"])[target]
    for name, offset in [("previous", -30), ("next", 30)]:
        keys = pd.MultiIndex.from_arrays(
            [data["nodeid"], data["time"] + offset]
        )
        inputs[f"{name}_reading"] = readings.reindex(keys).to_numpy()

    return inputs


def target_inputs(masked, targets, feature):
    # Answer rows identify which values to predict. Their true_value column
    # provides training/validation labels but never used as a predictor.
    features = make_features(masked, feature)
    return targets.merge(features, on=["time", "nodeid"], how="left", validate="one_to_one")


def main():
    parser = argparse.ArgumentParser(description="Random Forest sensor reconstruction")
    parser.add_argument("--sample-size", type=int, default=10_000)
    parser.add_argument("--save-predictions", action="store_true",
                        help="overwrite test predictions after choosing a model")
    args = parser.parse_args()
    if args.sample_size < 1:
        parser.error("--sample-size must be positive")

    train_masked = pd.read_csv(MASKED_DIR / "train_masked.csv")
    val_masked = pd.read_csv(MASKED_DIR / "val_masked.csv")
    test_masked = pd.read_csv(MASKED_DIR / "test_masked.csv") if args.save_predictions else None

    train_answers = pd.read_csv(MASKED_DIR / "train_answers.csv")
    val_answers = pd.read_csv(MASKED_DIR / "val_answers.csv")
    # Read only target locations and IDs, never held-out test values.
    test_targets = (
        pd.read_csv(MASKED_DIR / "test_answers.csv",
                    usecols=["target_id", "time", "nodeid", "feature"])
        if args.save_predictions else None
    )

    # Divide RMSE by each measurement's training-set standard deviation
    # so errors for temperature, humidity, light, and voltage are comparable.
    train_std = pd.read_csv(
        ROOT / "data/processed/train.csv", usecols=FEATURES
    ).std()

    all_predictions = []
    start = time.perf_counter()

    for feature in FEATURES:
        print(f"\nTraining Random Forest for {feature}...")
        train = target_inputs(
            train_masked, train_answers[train_answers["feature"] == feature], feature
        )
        val = target_inputs(
            val_masked, val_answers[val_answers["feature"] == feature], feature
        )
        if args.save_predictions:
            test = target_inputs(
                test_masked, test_targets[test_targets["feature"] == feature], feature
            )

        # Limit training rows for speed; the fixed seed makes runs repeatable.
        # Compare 10k and 30k using validation scores before selecting one.
        if len(train) > args.sample_size:
            train = train.sample(n=args.sample_size, random_state=RANDOM_STATE)

        columns = [
            "time", "nodeid", "x", "y",
            *[name for name in FEATURES if name != feature],
            "previous_reading", "next_reading",
        ]
        # Some predictors are missing. Fill them with medians computed from
        # training data only, avoiding information from validation/test rows.
        # Fall back to zero if an entire training column is missing.
        medians = train[columns].median().fillna(0)
        x_train = train[columns].fillna(medians)
        x_val = val[columns].fillna(medians)
        if args.save_predictions:
            x_test = test[columns].fillna(medians)

        # Use 50 trees for a manageable runtime. Depth and minimum leaf size
        # constrain tree complexity; all CPU cores can train trees in parallel.
        model = RandomForestRegressor(
            n_estimators=50,
            max_depth=20,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        model.fit(x_train, train["true_value"])

        # Evaluate on held-out validation labels; they do not train the model.
        val_predictions = model.predict(x_val)
        errors = val_predictions - val["true_value"].to_numpy()
        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors ** 2))
        nrmse = rmse / train_std[feature]
        print(
            f"  Training targets: {len(train):,}; "
            f"validation MAE: {mae:.6f}, RMSE: {rmse:.6f}, NRMSE: {nrmse:.6f}"
        )

        if args.save_predictions:
            all_predictions.append(pd.DataFrame({
                "target_id": test["target_id"].to_numpy(),
                "prediction": model.predict(x_test),
            }))

    if args.save_predictions:
        # The shared benchmark expects one prediction for each target_id.
        predictions = pd.concat(all_predictions).sort_values("target_id")
        if len(predictions) != len(test_targets) or predictions["target_id"].duplicated().any():
            raise ValueError("Missing or duplicate test predictions")
        if not np.isfinite(predictions["prediction"]).all():
            raise ValueError("Nonfinite test predictions")

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        output = RESULTS_DIR / "predictions.csv"
        predictions.to_csv(output, index=False)
        print(f"\nSaved {len(predictions):,} predictions to {output}")
    else:
        print("\nValidation only; existing test predictions were not changed.")
    print(f"Total runtime: {time.perf_counter() - start:.1f} seconds")


if __name__ == "__main__":
    main()
