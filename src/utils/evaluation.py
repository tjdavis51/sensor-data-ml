import os
import numpy as np
import pandas as pd


MODEL_NAME = "gru"

DATASET = "test"

MASKED_DIR = "data/masked"

TRAIN_PATH = "data/processed/train.csv"

MODELS_DIR = "src/models"

MEASUREMENT_COLUMNS = ["temperature", "humidity", "light", "voltage"]


def load_data(model_name, dataset):

    answers_path = os.path.join(MASKED_DIR, f"{dataset}_answers.csv")

    results_dir = os.path.join(MODELS_DIR, model_name, "results")

    predictions_path = os.path.join(results_dir, "predictions.csv")

    answers_df = pd.read_csv(answers_path)

    predictions_df = pd.read_csv(predictions_path)

    print(f"Loaded {len(answers_df):,} targets")
    print(f"Loaded {len(predictions_df):,} predictions")

    return answers_df, predictions_df, results_dir


def validate_predictions(answers_df, predictions_df):

    required_columns = ["target_id", "prediction"]

    missing_columns = [
        column for column in required_columns
        if column not in predictions_df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing prediction columns: {missing_columns}")

    duplicate_count = predictions_df["target_id"].duplicated().sum()

    if duplicate_count > 0:
        raise ValueError(f"Found {duplicate_count:,} duplicate target IDs")

    results_df = answers_df.merge(
        predictions_df,
        on="target_id",
        how="left",
        indicator=True,
        validate="one_to_one"
    )

    missing_count = (results_df["_merge"] == "left_only").sum()

    if missing_count > 0:
        raise ValueError(f"Missing predictions for {missing_count:,} targets")

    unknown_ids = ~predictions_df["target_id"].isin(answers_df["target_id"])

    unknown_count = unknown_ids.sum()

    if unknown_count > 0:
        raise ValueError(f"Found {unknown_count:,} unknown target IDs")

    results_df["prediction"] = pd.to_numeric(
        results_df["prediction"],
        errors="coerce"
    )

    invalid_predictions = ~np.isfinite(results_df["prediction"].to_numpy())

    invalid_count = invalid_predictions.sum()

    if invalid_count > 0:
        raise ValueError(f"Found {invalid_count:,} invalid predictions")

    results_df = results_df.drop(columns=["_merge"])

    return results_df


def get_training_std():

    train_df = pd.read_csv(TRAIN_PATH)

    training_std = train_df[MEASUREMENT_COLUMNS].std()

    return training_std


def evaluate_predictions(results_df, training_std):

    metrics = []

    for column in MEASUREMENT_COLUMNS:

        feature_df = results_df[results_df["feature"] == column]

        true_values = feature_df["true_value"].to_numpy()

        predicted_values = feature_df["prediction"].to_numpy()

        errors = predicted_values - true_values

        mae = np.mean(np.abs(errors))

        rmse = np.sqrt(np.mean(errors ** 2))

        nrmse = rmse / training_std[column]

        metrics.append({
            "feature": column,
            "count": len(feature_df),
            "mae": mae,
            "rmse": rmse,
            "nrmse": nrmse,
        })

    metrics_df = pd.DataFrame(metrics)

    overall_score = metrics_df["nrmse"].mean()

    return metrics_df, overall_score


def print_results(model_name, dataset, metrics_df, overall_score):

    print(f"\n{model_name.upper()} Evaluation Results ({dataset.upper()}):\n")

    for _, row in metrics_df.iterrows():

        print(f"{row['feature'].capitalize()} ({row['count']:,} targets):")

        print(f"  MAE:   {row['mae']:.6f}")
        print(f"  RMSE:  {row['rmse']:.6f}")
        print(f"  NRMSE: {row['nrmse']:.6f}")
        print()

    print(f"Overall NRMSE: {overall_score:.6f}")


def save_results(metrics_df, overall_score, results_dir):

    os.makedirs(results_dir, exist_ok=True)

    output_path = os.path.join(results_dir, "evaluation.csv")

    overall_row = pd.DataFrame([{
        "feature": "overall",
        "count": metrics_df["count"].sum(),
        "mae": np.nan,
        "rmse": np.nan,
        "nrmse": overall_score,
    }])

    output_df = pd.concat([metrics_df, overall_row], ignore_index=True)

    output_df.to_csv(output_path, index=False)

    print(f"\nSaved evaluation results to {output_path}")


def main():

    print("\nEvaluating Model Predictions:\n")

    print(f"Model: {MODEL_NAME}")
    print(f"Dataset: {DATASET}")

    answers_df, predictions_df, results_dir = load_data(MODEL_NAME, DATASET)

    results_df = validate_predictions(answers_df, predictions_df)

    training_std = get_training_std()

    metrics_df, overall_score = evaluate_predictions(results_df, training_std)

    print_results(MODEL_NAME, DATASET, metrics_df, overall_score)

    save_results(metrics_df, overall_score, results_dir)

    print("\nEvaluation complete")


if __name__ == "__main__":
    main()
