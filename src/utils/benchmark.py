# runs the models and collects the results

# for testing
# each algorithm needs the same input: the masked dataset
# the task is the same for each: predict the missing measurements in the target period
# then the outputs need to match: a single CSV containing the predictions

# compare the predictions from all available models
# use evaluation.py to calculate consistent metrics for each model
# skip models that do not have a predictions file
# save the comparison results to a csv file

import os
import pandas as pd

# import the shared evaluation functions
from evaluation import (
    load_data,
    validate_predictions,
    get_training_std,
    evaluate_predictions,
)


# dataset split to benchmark
DATASET = "test"

# directory containing the model implementations
MODELS_DIR = "src/models"

# directory where the benchmark results will be saved
OUTPUT_DIR = "results"

# models to include in the benchmark
MODEL_NAMES = [
    "linear_regression",
    "random_forest",
    "mlp",
    "gru",
    "gnn",
]


# function to check if a model has predictions available
def has_predictions(model_name):

    # create the path to the model's predictions file
    predictions_path = os.path.join(
        MODELS_DIR,
        model_name,
        "results",
        "predictions.csv"
    )

    # return whether the file exists
    return os.path.isfile(predictions_path)


# function to evaluate a single model
def benchmark_model(model_name, training_std):

    # load the answer key and model predictions
    answers_df, predictions_df, _ = load_data(model_name, DATASET)

    # validate and match the predictions
    results_df = validate_predictions(answers_df, predictions_df)

    # calculate the evaluation metrics
    metrics_df, overall_score = evaluate_predictions(
        results_df,
        training_std
    )

    # create a dictionary to store the model's results
    results = {
        "model": model_name,
        "overall_nrmse": overall_score,
    }

    # add the evaluation metrics for each measurement type
    for _, row in metrics_df.iterrows():

        # get the measurement name
        feature = row["feature"]

        # save the MAE, RMSE, and NRMSE
        results[f"{feature}_mae"] = row["mae"]
        results[f"{feature}_rmse"] = row["rmse"]
        results[f"{feature}_nrmse"] = row["nrmse"]

    # return the model's benchmark results
    return results


# function to print the benchmark comparison
def print_results(benchmark_df):

    print("\nBenchmark Results:\n")

    # select the most important metrics for the comparison
    columns = [
        "model",
        "temperature_rmse",
        "humidity_rmse",
        "light_rmse",
        "voltage_rmse",
        "overall_nrmse",
    ]

    # print the comparison table
    print(benchmark_df[columns].to_string(index=False))

    print()


# function to save the benchmark results
def save_results(benchmark_df):

    # create the output directory if needed
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # create the path for the benchmark results
    output_path = os.path.join(
        OUTPUT_DIR,
        f"benchmark_{DATASET}.csv"
    )

    # save the benchmark results
    benchmark_df.to_csv(output_path, index=False)

    # print a success message
    print(f"Saved benchmark results to {output_path}")


# main function to benchmark all available models
def main():

    print("\nBenchmarking Model Predictions:\n")

    # print the selected dataset
    print(f"Dataset: {DATASET.upper()}")

    # calculate the training standard deviations once
    training_std = get_training_std()

    # create a list to store the results from each model
    benchmark_results = []

    # go through each model
    for model_name in MODEL_NAMES:

        # check if the model has predictions available
        if not has_predictions(model_name):

            # print a message and skip the model
            print(f"\nSkipping {model_name}: predictions.csv not found")
            continue

        # print the model being evaluated
        print(f"\nEvaluating {model_name}:")

        # evaluate the model's predictions
        results = benchmark_model(model_name, training_std)

        # save the model's results
        benchmark_results.append(results)

    # stop if no models have predictions available
    if not benchmark_results:
        print("\nNo model predictions found")
        return

    # create a dataframe containing the benchmark results
    benchmark_df = pd.DataFrame(benchmark_results)

    # sort the models by overall normalized RMSE
    benchmark_df = benchmark_df.sort_values("overall_nrmse")

    # reset the row indices
    benchmark_df = benchmark_df.reset_index(drop=True)

    # print the benchmark comparison
    print_results(benchmark_df)

    # save the benchmark results
    save_results(benchmark_df)

    # print a success message
    print("\nBenchmark complete")


# run main() only when this file is executed directly
if __name__ == "__main__":
    main()