# calculate the metrics from prediction and ground truth

# evaluate a model's predictions against the masked dataset answer key
# calculate MAE, RMSE, and NRMSE for each measurement type
# use the original training dataset to normalize RMSE
# save the evaluation results in the model's results folder

import os
import numpy as np
import pandas as pd


# name of the model to evaluate
MODEL_NAME = "gru"

# dataset split to evaluate
DATASET = "test"

# directory containing the answer keys
MASKED_DIR = "data/masked"

# path to the original training dataset
TRAIN_PATH = "data/processed/train.csv"

# directory containing the model implementations
MODELS_DIR = "src/models"

# measurement columns to evaluate
MEASUREMENT_COLUMNS = ["temperature", "humidity", "light", "voltage"]


# function to load the answer key and predictions
def load_data(model_name, dataset):

    # create the path to the answer key
    answers_path = os.path.join(MASKED_DIR, f"{dataset}_answers.csv")

    # create the path to the model's results folder
    results_dir = os.path.join(MODELS_DIR, model_name, "results")

    # create the path to the predictions file
    predictions_path = os.path.join(results_dir, "predictions.csv")

    # load the answer key
    answers_df = pd.read_csv(answers_path)

    # load the model predictions
    predictions_df = pd.read_csv(predictions_path)

    # print the number of predictions and targets
    print(f"Loaded {len(answers_df):,} targets")
    print(f"Loaded {len(predictions_df):,} predictions")

    # return the dataframes and results directory
    return answers_df, predictions_df, results_dir


# function to validate and match the predictions
def validate_predictions(answers_df, predictions_df):

    # check that the predictions contain the required columns
    required_columns = ["target_id", "prediction"]

    # identify any missing columns
    missing_columns = [
        column for column in required_columns
        if column not in predictions_df.columns
    ]

    # stop if required columns are missing
    if missing_columns:
        raise ValueError(f"Missing prediction columns: {missing_columns}")

    # check for duplicate prediction IDs
    duplicate_count = predictions_df["target_id"].duplicated().sum()

    # stop if duplicate IDs are found
    if duplicate_count > 0:
        raise ValueError(f"Found {duplicate_count:,} duplicate target IDs")

    # match predictions to answers using target_id
    results_df = answers_df.merge(
        predictions_df,
        on="target_id",
        how="left",
        indicator=True,
        validate="one_to_one"
    )

    # count targets without predictions
    missing_count = (results_df["_merge"] == "left_only").sum()

    # stop if any targets are missing predictions
    if missing_count > 0:
        raise ValueError(f"Missing predictions for {missing_count:,} targets")

    # identify prediction IDs that do not exist in the answer key
    unknown_ids = ~predictions_df["target_id"].isin(answers_df["target_id"])

    # count unknown prediction IDs
    unknown_count = unknown_ids.sum()

    # stop if unknown IDs are found
    if unknown_count > 0:
        raise ValueError(f"Found {unknown_count:,} unknown target IDs")

    # convert predictions to numeric values
    results_df["prediction"] = pd.to_numeric(
        results_df["prediction"],
        errors="coerce"
    )

    # check that all predictions are finite
    invalid_predictions = ~np.isfinite(results_df["prediction"].to_numpy())

    # count invalid predictions
    invalid_count = invalid_predictions.sum()

    # stop if any predictions are invalid
    if invalid_count > 0:
        raise ValueError(f"Found {invalid_count:,} invalid predictions")

    # remove the merge indicator column
    results_df = results_df.drop(columns=["_merge"])

    # return the validated results
    return results_df


# function to calculate training standard deviations
def get_training_std():

    # load the original training dataset
    train_df = pd.read_csv(TRAIN_PATH)

    # calculate the standard deviation for each measurement type
    training_std = train_df[MEASUREMENT_COLUMNS].std()

    # return the standard deviations
    return training_std


# function to calculate evaluation metrics
def evaluate_predictions(results_df, training_std):

    # create a list to store the evaluation results
    metrics = []

    # go through each measurement type
    for column in MEASUREMENT_COLUMNS:

        # select predictions for the current measurement
        feature_df = results_df[results_df["feature"] == column]

        # get the original values
        true_values = feature_df["true_value"].to_numpy()

        # get the predicted values
        predicted_values = feature_df["prediction"].to_numpy()

        # calculate the prediction errors
        errors = predicted_values - true_values

        # calculate mean absolute error
        mae = np.mean(np.abs(errors))

        # calculate root mean squared error
        rmse = np.sqrt(np.mean(errors ** 2))

        # normalize RMSE using the training standard deviation
        nrmse = rmse / training_std[column]

        # save the evaluation metrics
        metrics.append({
            "feature": column,
            "count": len(feature_df),
            "mae": mae,
            "rmse": rmse,
            "nrmse": nrmse,
        })

    # create a dataframe containing the metrics
    metrics_df = pd.DataFrame(metrics)

    # calculate the average normalized RMSE
    overall_score = metrics_df["nrmse"].mean()

    # return the metrics and overall score
    return metrics_df, overall_score


# function to print the evaluation results
def print_results(model_name, dataset, metrics_df, overall_score):

    print(f"\n{model_name.upper()} Evaluation Results ({dataset.upper()}):\n")

    # print the results for each measurement type
    for _, row in metrics_df.iterrows():

        # print the measurement name and number of targets
        print(f"{row['feature'].capitalize()} ({row['count']:,} targets):")

        # print the evaluation metrics
        print(f"  MAE:   {row['mae']:.6f}")
        print(f"  RMSE:  {row['rmse']:.6f}")
        print(f"  NRMSE: {row['nrmse']:.6f}")
        print()

    # print the overall normalized score
    print(f"Overall NRMSE: {overall_score:.6f}")


# function to save the evaluation results
def save_results(metrics_df, overall_score, results_dir):

    # create the results directory if needed
    os.makedirs(results_dir, exist_ok=True)

    # create the path for the evaluation results
    output_path = os.path.join(results_dir, "evaluation.csv")

    # add an overall score row to the results
    overall_row = pd.DataFrame([{
        "feature": "overall",
        "count": metrics_df["count"].sum(),
        "mae": np.nan,
        "rmse": np.nan,
        "nrmse": overall_score,
    }])

    # combine the feature metrics and overall score
    output_df = pd.concat([metrics_df, overall_row], ignore_index=True)

    # save the results to csv
    output_df.to_csv(output_path, index=False)

    # print a success message
    print(f"\nSaved evaluation results to {output_path}")


# main function to evaluate the selected model
def main():

    print("\nEvaluating Model Predictions:\n")

    # print the selected model and dataset
    print(f"Model: {MODEL_NAME}")
    print(f"Dataset: {DATASET}")

    # load the answer key and predictions
    answers_df, predictions_df, results_dir = load_data(MODEL_NAME, DATASET)

    # validate and match the predictions
    results_df = validate_predictions(answers_df, predictions_df)

    # calculate the training standard deviations
    training_std = get_training_std()

    # calculate the evaluation metrics
    metrics_df, overall_score = evaluate_predictions(results_df, training_std)

    # print the evaluation results
    print_results(MODEL_NAME, DATASET, metrics_df, overall_score)

    # save the evaluation results
    save_results(metrics_df, overall_score, results_dir)

    # print a success message
    print("\nEvaluation complete")


# run main() only when this file is executed directly
if __name__ == "__main__":
    main()