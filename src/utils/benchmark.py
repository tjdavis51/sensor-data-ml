import os
import pandas as pd

from evaluation import (
    load_data,
    validate_predictions,
    get_training_std,
    evaluate_predictions,
)


DATASET = "test"

MODELS_DIR = "src/models"

OUTPUT_DIR = "results"

MODEL_NAMES = [
    "linear_regression",
    "random_forest",
    "neural_network",
    "gru",
    "gnn",
]


def has_predictions(model_name):

    predictions_path = os.path.join(
        MODELS_DIR,
        model_name,
        "results",
        "predictions.csv"
    )

    return os.path.isfile(predictions_path)


def benchmark_model(model_name, training_std):

    answers_df, predictions_df, _ = load_data(model_name, DATASET)

    results_df = validate_predictions(answers_df, predictions_df)

    metrics_df, overall_score = evaluate_predictions(
        results_df,
        training_std
    )

    results = {
        "model": model_name,
        "overall_nrmse": overall_score,
    }

    for _, row in metrics_df.iterrows():

        feature = row["feature"]

        results[f"{feature}_mae"] = row["mae"]
        results[f"{feature}_rmse"] = row["rmse"]
        results[f"{feature}_nrmse"] = row["nrmse"]

    return results


def print_results(benchmark_df):

    print("\nBenchmark Results:\n")

    columns = [
        "model",
        "temperature_rmse",
        "humidity_rmse",
        "light_rmse",
        "voltage_rmse",
        "overall_nrmse",
    ]

    print(benchmark_df[columns].to_string(index=False))

    print()


def save_results(benchmark_df):

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    output_path = os.path.join(
        OUTPUT_DIR,
        f"benchmark_{DATASET}.csv"
    )

    benchmark_df.to_csv(output_path, index=False)

    print(f"Saved benchmark results to {output_path}")


def main():

    print("\nBenchmarking Model Predictions:\n")

    print(f"Dataset: {DATASET.upper()}")

    training_std = get_training_std()

    benchmark_results = []

    for model_name in MODEL_NAMES:

        if not has_predictions(model_name):

            print(f"\nSkipping {model_name}: predictions.csv not found")
            continue

        print(f"\nEvaluating {model_name}:")

        results = benchmark_model(model_name, training_std)

        benchmark_results.append(results)

    if not benchmark_results:
        print("\nNo model predictions found")
        return

    benchmark_df = pd.DataFrame(benchmark_results)

    benchmark_df = benchmark_df.sort_values("overall_nrmse")

    benchmark_df = benchmark_df.reset_index(drop=True)

    print_results(benchmark_df)

    save_results(benchmark_df)

    print("\nBenchmark complete")


if __name__ == "__main__":
    main()
