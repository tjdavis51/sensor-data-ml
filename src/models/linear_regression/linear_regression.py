# linear regression algorithm for predicting masked sensor readings
# data prepared using preprocess.py, split_dataset.py, and masking.py
from pathlib import Path
import argparse
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler


# (parents[0] = linear_regression, [1] = models, [2] = src, [3] = sensor-data-ml)
ROOT = Path(__file__).resolve().parents[3]
MASKED_DIR = ROOT / "data/masked" # folder with the masked inputs and answer keys from masking.py
RESULTS_DIR = ROOT / "src/models/linear_regression/results" # where predictions.csv is saved
FEATURES = ["temperature", "humidity", "light", "voltage"] # data measurements we have
STEP = 30 # seconds between synchronized readings, aligns with preprocess.py's binning
SECONDS_PER_DAY = 24 * 60 * 60 # 86400
MIN_GROUP_ROWS = 200 # smaller training groups use the pooled model


def nearest_observed(masked, target, rows, max_gap):
    # nearest observed reading of the same sensor (nodeid) before and after each target row
    # looks a maximum of max_gap STEPs (30s) each way. returns the readings and how many steps away
    # they were, or NaN if nothing is observed within the period

    # table with 1 row per 30 second timestamp, and 1 column per sensor, NaN marked as missing
    wide = masked.pivot_table(index="time", columns="nodeid", values=target)

    # full time grid so 1 row = 1 30 second step
    grid = np.arange(int(wide.index.min()), int(wide.index.max()) + STEP, STEP)
    wide = wide.reindex(grid)

    # row number (step index) repeated across every sensor column
    steps = np.arange(len(wide))[:, None] * np.ones((1, wide.shape[1]))
    # keep the step index if a reading exists, and NaN elsewhere
    positions = pd.DataFrame(
        np.where(wide.notna().to_numpy(), steps, np.nan),
        index=wide.index, columns=wide.columns,
    )
    # carry the last seen step index forward / next seen index backward. shifting by one
    # makes it always strictly before or after the current step (never the target itself)
    before = positions.ffill().shift(1).to_numpy()    # last observed step before t
    after = positions.bfill().shift(-1).to_numpy()    # next observed step after t

    # locate each target row in the grid: r = row (time step), c = column (sensor)
    r = wide.index.get_indexer(rows["time"])
    c = wide.columns.get_indexer(rows["nodeid"])
    values = wide.to_numpy()
    prev_idx, next_idx = before[r, c], after[r, c]
    # how many steps away the neighbors are (NaN if none exists)
    gap_prev, gap_next = r - prev_idx, next_idx - r

    # look up the neighbor's value, but only keep it if it is within max_gap steps
    # nan_to_num keeps the index valid when there is no neighbor; those get masked to NaN
    prev_val = np.where(gap_prev <= max_gap,
                        values[np.nan_to_num(prev_idx).astype(int), c], np.nan)
    next_val = np.where(gap_next <= max_gap,
                        values[np.nan_to_num(next_idx).astype(int), c], np.nan)
    return (prev_val, next_val,
            np.where(gap_prev <= max_gap, gap_prev, np.nan),
            np.where(gap_next <= max_gap, gap_next, np.nan))


def target_inputs(masked, targets, feature, max_gap):
    # one row per target, with predictors that do not include the hidden value
    # the other three measurements at the same moment can help explain the missing one
    other = [name for name in FEATURES if name != feature]
    base = masked[["time", "nodeid", "x", "y"] + other]
    # attach each target's inputs; one_to_one because targets were already filtered to one feature
    rows = targets.merge(base, on=["time", "nodeid"], how="left", validate="one_to_one")

    # smooth daily cycle (raw time would extrapolate: test is later than train)
    angle = 2 * np.pi * (rows["time"] % SECONDS_PER_DAY) / SECONDS_PER_DAY
    rows["tod_sin"], rows["tod_cos"] = np.sin(angle), np.cos(angle)

    # average of this measurement across all sensors at the same timestamp
    rows["time_mean"] = rows["time"].map(masked.groupby("time")[feature].mean())

    # same-sensor neighbors in time, plus how far away they are
    prev_val, next_val, gap_prev, gap_next = nearest_observed(masked, feature, rows, max_gap)
    rows["prev_val"], rows["next_val"] = prev_val, next_val
    rows["gap_prev"], rows["gap_next"] = gap_prev, gap_next
    # availability flags: both sides found / exactly one side found (xor); neither = neither flag
    rows["both"] = rows["prev_val"].notna() & rows["next_val"].notna()
    rows["one"] = rows["prev_val"].notna() ^ rows["next_val"].notna()

    # interpolation features (a linear model can use these directly)
    rows["mean_nb"] = rows[["prev_val", "next_val"]].mean(axis=1)
    # time-weighted interpolation: the closer neighbor gets the larger weight
    weight = rows["gap_next"] / (rows["gap_prev"] + rows["gap_next"])
    rows["tw_interp"] = rows["prev_val"] * weight + rows["next_val"] * (1 - weight)
    return rows


def pattern_columns(feature):
    # which input columns each availability pattern's model uses
    other = [name for name in FEATURES if name != feature]
    shared = ["x", "y", "tod_sin", "tod_cos", "time_mean", *other] # always available
    return {
        "both": shared + ["tw_interp", "prev_val", "next_val", "gap_prev", "gap_next"],
        "one": shared + ["mean_nb", "gap_prev", "gap_next"],
        "none": shared, # no same-sensor neighbors, rely on other sensors/measurements/time of day
    }


def pattern_masks(df):
    # boolean row masks that split targets by how many same-sensor neighbors were found
    return {"both": df["both"], "one": df["one"], "none": ~(df["both"] | df["one"])}


def make_model(kind, alpha):
    # ols = ordinary least squares, ridge = least squares with an L2 penalty (alpha)
    return LinearRegression() if kind == "ols" else Ridge(alpha=alpha)


def fit_group(train_rows, columns, kind, alpha):
    # median-impute and scale using training rows only, then fit
    medians = train_rows[columns].median().fillna(0) # fill value for any missing inputs (0 if a column is all NaN)
    x = train_rows[columns].fillna(medians)
    scaler = StandardScaler().fit(x) # mean 0, std 1; learned from training data only (no leakage)
    model = make_model(kind, alpha).fit(scaler.transform(x), train_rows["true_value"])
    # keep everything needed to apply the same steps to new rows later
    return {"columns": columns, "medians": medians, "scaler": scaler, "model": model}


def predict_group(fitted, rows):
    # apply the same imputation + scaling learned in fit_group, then predict
    x = rows[fitted["columns"]].fillna(fitted["medians"])
    return fitted["model"].predict(fitted["scaler"].transform(x))


def fit_feature(train, feature, kind, alpha):
    columns = pattern_columns(feature)
    pooled = fit_group(train, columns["none"], kind, alpha) # fallback model using all training rows
    fitted = {}
    for name, mask in pattern_masks(train).items():
        group = train[mask]
        # one model per availability pattern, unless the group is too small to fit reliably
        fitted[name] = (fit_group(group, columns[name], kind, alpha)
                        if len(group) >= MIN_GROUP_ROWS else pooled)
    return fitted


def predict_feature(fitted, rows):
    # route each row to the model for its availability pattern
    predictions = pd.Series(np.nan, index=rows.index)
    for name, mask in pattern_masks(rows).items():
        if mask.any():
            predictions[mask] = predict_group(fitted[name], rows[mask])
    return predictions


def main():
    # set up parser and add arguments, save to args array
    parser = argparse.ArgumentParser(description="Linear regression sensor reconstruction")
    parser.add_argument("--model", choices=["ols", "ridge"], default="ols")
    parser.add_argument("--alpha", type=float, default=1.0,
                            help="ridge penalty (only used with --model ridge)")
    parser.add_argument("--max-gap", type=int, default=20,
                            help="how many 30 s steps to search for same-sensor neighbors")
    parser.add_argument("--save-predictions", action="store_true",
                            help="write test predictions for the shared benchmark")
    args = parser.parse_args()

    # load the csv files created by preprocess.py, split_dataset.py, and masking.py
    train_masked = pd.read_csv(MASKED_DIR / "train_masked.csv")
    val_masked = pd.read_csv(MASKED_DIR / "val_masked.csv")
    test_masked = pd.read_csv(MASKED_DIR / "test_masked.csv") if args.save_predictions else None

    train_answers = pd.read_csv(MASKED_DIR / "train_answers.csv")
    val_answers = pd.read_csv(MASKED_DIR / "val_answers.csv")

    # only read location and ID of target test answers
    test_targets = (
            pd.read_csv(MASKED_DIR / "test_answers.csv",
                        usecols=["target_id", "time", "nodeid", "feature"])
            if args.save_predictions else None
        )

    train_std = pd.read_csv(ROOT / "data/processed/train.csv", usecols=FEATURES).std()

    all_predictions, scores = [], []
    start = time.perf_counter()

    for feature in FEATURES:
        print(f"\nTraining {args.model} regression for {feature}...")
        # create a row per hidden training value (inputs and true value)
        train = target_inputs(train_masked, train_answers[train_answers["feature"] == feature],
                                    feature, args.max_gap)
        # same process for validation set, used to assess the model
        val = target_inputs(val_masked, val_answers[val_answers["feature"] == feature],
                                    feature, args.max_gap)
        # fit the 3 linear models (both, one, or no neighbors) on only the training rows
        fitted = fit_feature(train, feature, args.model, args.alpha)

        val_pred = predict_feature(fitted, val) # get predicted values
        errors = val_pred - val["true_value"] # get difference from predicted to true
        mae = errors.abs().mean() # mean absolute error
        rmse = np.sqrt((errors ** 2).mean()) # root mean squared error

        # normalize by the training std, this way all features share the same scale
        nrmse = rmse / train_std[feature]
        scores.append(nrmse)    # saved for the overall average below
        print(f"  Training targets: {len(train):,}; "
              f"validation MAE: {mae:.6f}, RMSE: {rmse:.6f}, NRMSE: {nrmse:.6f}")

        # analyze error by how many same-sensor neighbor readings (before/after in time) were found
        for name, mask in pattern_masks(val).items():
            if mask.any():                        # skip any groups with no rows
                group_rmse = np.sqrt((errors[mask] ** 2).mean())
                print(f"    {name:>4} neighbors: {mask.mean():6.1%} of targets, "
                      f"NRMSE {group_rmse / train_std[feature]:.4f}")

        if args.save_predictions:
            # test_targets has no true values, so the model never sees test answers
            test = target_inputs(test_masked, test_targets[test_targets["feature"] == feature],
                                 feature, args.max_gap)
            # keep target_id so evaluation.py can match predictions to answers
            all_predictions.append(pd.DataFrame({
                "target_id": test["target_id"].to_numpy(),
                "prediction": predict_feature(fitted, test).to_numpy(),
            }))
    # overall score = average NRMSE across the four features (same as evaluation.py)
    print(f"\nValidation overall NRMSE (mean of 4 features): {np.mean(scores):.6f}")

    if args.save_predictions:
        # stack the four features' predictions and order them by target_id
        predictions = pd.concat(all_predictions).sort_values("target_id")

        # sanity checks: exactly one prediction per target, no repeats
        if len(predictions) != len(test_targets) or predictions["target_id"].duplicated().any():
            raise ValueError("Missing or duplicate test predictions")
        # sanity check: no NaN or infinite predictions
        if not np.isfinite(predictions["prediction"]).all():
            raise ValueError("Nonfinite test predictions")

        # write the file that evaluation.py reads
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        output = RESULTS_DIR / "predictions.csv"
        predictions.to_csv(output, index=False)
        print(f"Saved {len(predictions):,} predictions to {output}")
    else:
        print("Validation only; no test predictions were written.")
    print(f"Total runtime: {time.perf_counter() - start:.1f} seconds")


if __name__ == "__main__":
    main()