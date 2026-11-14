import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler

import pandas as pd
import os
import numpy as np
import argparse
INPUT_DIR = "data/processed"

TIME_STEP = 30
MEASUREMENT_COLUMNS = ["temperature", "humidity", "light", "voltage"]
NUM_NEIGHBORS = 4

CHECKPOINT_DIR = "src/models/gru/checkpoints"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pt")

OUTPUT_DIR = "src/models/gru/results"
PREDICTIONS_PATH = os.path.join(OUTPUT_DIR, "predictions.csv")

DEVICE = torch.device("cpu")

print(f"Using device: {DEVICE}")


def load_dataset(name):

    path = os.path.join(INPUT_DIR, f"{name}.csv")

    df = pd.read_csv(path)

    print(f"Loaded {len(df):,} {name} observations")

    return df

def build_sensor_arrays(df, sensor_ids=None):

    if sensor_ids is None:
        sensor_ids = np.sort(df["nodeid"].unique())

    start_time = int(df["time"].min())
    end_time = int(df["time"].max())

    times = np.arange(
        start_time,
        end_time + TIME_STEP,
        TIME_STEP
    )

    assert (df["time"] % TIME_STEP == 0).all()

    indexed_df = df.set_index(["time", "nodeid"])

    full_index = pd.MultiIndex.from_product(
        [times, sensor_ids],
        names=["time", "nodeid"]
    )

    measurements = indexed_df[MEASUREMENT_COLUMNS]
    measurements = measurements.reindex(full_index)

    values = measurements.to_numpy(dtype=np.float32)


    values = values.reshape(
        len(times),
        len(sensor_ids),
        len(MEASUREMENT_COLUMNS)
    )

    availability = ~np.isnan(values)

    print(f"Sensor array shape: {values.shape}")

    return values, availability, times, sensor_ids

def squared_eu_distance(loc1, loc2):
    return (loc1[0] - loc2[0]) ** 2 + (loc1[1] - loc2[1]) ** 2

def find_nearest_neighbors(df, sensor_ids, num_neighbors):
    sensors = {}

    for sensor_id in sensor_ids:

        sensor = df[df["nodeid"] == sensor_id].iloc[0]

        position = (sensor["x"], sensor["y"])

        sensors[int(sensor_id)] = {
            "position": position,
            "neighbors": [None] * num_neighbors,
            "distances": [float("inf")] * num_neighbors
        }

    for key_id in sensors:
        for comparison_id in sensors:

            if key_id == comparison_id:
                continue

            dist = squared_eu_distance(
                sensors[key_id]["position"],
                sensors[comparison_id]["position"]
            )

            for i in range(num_neighbors):
                if dist < sensors[key_id]["distances"][i]:

                    sensors[key_id]["distances"].insert(i, dist)
                    sensors[key_id]["neighbors"].insert(i, comparison_id)

                    sensors[key_id]["distances"].pop()
                    sensors[key_id]["neighbors"].pop()

                    break

    id_to_index = {
        int(sensor_id): i
        for i, sensor_id in enumerate(sensor_ids)
    }

    neighbors = []

    for sensor_id in sensor_ids:
        sensor_id = int(sensor_id)

        id_with_neighbors = [sensor_id] + sensors[sensor_id]["neighbors"]

        index_with_neighbors = [
            id_to_index[neighbor_id]
            for neighbor_id in id_with_neighbors
        ]

        neighbors.append(index_with_neighbors)

    return np.array(neighbors, dtype=np.int32)

def fit_normalization(values):
    means = np.nanmean(values, axis=(0, 1))
    stds = np.nanstd(values, axis=(0, 1))

    stds[stds == 0] = 1.0

    return means, stds

def normalize_measurements(values, means, stds):
    normalized = (values - means) / stds

    normalized = np.nan_to_num(normalized, nan=0.0)

    return normalized


class SensorDataset(Dataset):

    def __init__(self, values, availability, neighbors, sequence_length=40, mask_probability=0.1):

        self.values = values
        self.availability = availability
        self.neighbors = neighbors
        self.sequence_length = sequence_length
        self.mask_probability = mask_probability

        self.num_times = len(values)
        self.num_sensors = len(neighbors)

    def __len__(self):
        windows_per_sensor = self.num_times - self.sequence_length + 1

        return windows_per_sensor * self.num_sensors

    def __getitem__(self, index):
        windows_per_sensor = self.num_times - self.sequence_length + 1

        sensor_index = index // windows_per_sensor
        start = index % windows_per_sensor
        end = start + self.sequence_length

        sensor_indices = self.neighbors[sensor_index]

        measurements = self.values[start:end, sensor_indices, :].copy()
        availability = self.availability[start:end, sensor_indices, :].copy()

        y = measurements[:, 0, :].copy()

        mask = (
            np.random.random(y.shape) < self.mask_probability
        ) & availability[:, 0, :]

        measurements[:, 0, :][mask] = 0.0
        availability[:, 0, :][mask] = False

        x = torch.cat([
            torch.tensor(measurements, dtype=torch.float32),
            torch.tensor(availability, dtype=torch.float32)
        ], dim=2)

        x = x.reshape(self.sequence_length, -1)

        return (
            x,
            torch.tensor(y, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.bool)
        )


class SensorGRU(nn.Module):

    def __init__(self):

        super().__init__()

        self.gru = nn.GRU(
            input_size=40,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        self.output = nn.Linear(256, 4)

    def forward(self, x):

        hidden_states, _ = self.gru(x)

        predictions = self.output(hidden_states)

        return predictions


def masked_mse_loss(predictions, targets, mask):
    if not mask.any().item():
        raise ValueError("No masked targets in batch")

    return torch.mean((predictions[mask] - targets[mask]) ** 2)

def train_model(dataset, val_data, means, stds, sensor_ids, neighbors,
                epochs=5, batch_size=64, learning_rate=0.001, samples_per_epoch=50000):
    
    model = SensorGRU().to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    sampler = RandomSampler(
        dataset,
        replacement=False,
        num_samples=min(samples_per_epoch, len(dataset))
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler
    )

    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()

        total_loss = 0.0
        total_batches = 0

        for x, y, mask in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            mask = mask.to(DEVICE)
            if not mask.any():
                continue

            optimizer.zero_grad()

            predictions = model(x)

            loss = masked_mse_loss(predictions, y, mask)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()
            total_batches += 1

        average_loss = total_loss / max(total_batches, 1)

        print(f"Epoch {epoch + 1}/{epochs} - Training MSE: {average_loss:.4f}")


        val_values, val_availability, val_times, val_answers = val_data

        val_loss = validate_model(
            model,
            val_values,
            val_availability,
            neighbors,
            val_times,
            sensor_ids,
            val_answers,
            means,
            stds
        )

        print(f"Epoch {epoch + 1}/{epochs} - Validation MSE: {val_loss:.4f}")

        if np.isfinite(val_loss) and val_loss < best_val_loss:
            best_val_loss = val_loss

            save_checkpoint(
                model, means, stds, sensor_ids, neighbors
            )

            print(f"Saved best model with validation MSE: {best_val_loss:.4f}")


    return model


def validate_model(model, values, availability, neighbors, times,
                   sensor_ids, answers, means, stds, batch_size=128):

    model.eval()

    id_to_index = {int(sensor_id): i for i, sensor_id in enumerate(sensor_ids)}
    time_to_index = {int(time): i for i, time in enumerate(times)}
    feature_to_index = {name: i for i, name in enumerate(MEASUREMENT_COLUMNS)}

    sequence_length = 40
    device = next(model.parameters()).device

    total_squared_error = 0.0
    total_predictions = 0

    with torch.inference_mode():
        for batch_start in range(0, len(answers), batch_size):
            batch = answers.iloc[batch_start:batch_start + batch_size]

            inputs = []
            target_values = []
            target_steps = []
            target_features = []

            for row in batch.itertuples(index=False):
                sensor_index = id_to_index[int(row.nodeid)]
                time_index = time_to_index[int(row.time)]
                feature_index = feature_to_index[row.feature]

                if availability[time_index, sensor_index, feature_index]:
                    raise ValueError(
                        f"Validation target is visible: "
                        f"time={row.time}, "
                        f"nodeid={row.nodeid}, "
                        f"feature={row.feature}"
                    )

                start = max(0, min(
                    time_index - sequence_length // 2,
                    len(times) - sequence_length
                ))
                end = start + sequence_length

                sensor_indices = neighbors[sensor_index]

                measurements = values[start:end, sensor_indices, :]
                flags = availability[start:end, sensor_indices, :]

                x = np.concatenate(
                    [measurements, flags.astype(np.float32)],
                    axis=2
                ).reshape(sequence_length, -1)

                inputs.append(x)

                target = (row.true_value - means[feature_index]) / stds[feature_index]

                target_values.append(target)
                target_steps.append(time_index - start)
                target_features.append(feature_index)

            x = torch.from_numpy(np.stack(inputs)).to(device)
            predictions = model(x)

            indices = torch.arange(len(batch), device=device)
            steps = torch.tensor(target_steps, device=device)
            features = torch.tensor(target_features, device=device)

            selected_predictions = predictions[indices, steps, features]

            targets = torch.tensor(target_values, dtype=torch.float32, device=device)
            squared_errors = (selected_predictions - targets) ** 2

            total_squared_error += squared_errors.sum().item()
            total_predictions += len(batch)

    return total_squared_error / total_predictions


def save_checkpoint(model, means, stds, sensor_ids, neighbors):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "means": torch.as_tensor(means).cpu(),
        "stds": torch.as_tensor(stds).cpu(),
        "sensor_ids": torch.as_tensor(sensor_ids).cpu(),
        "neighbors": torch.as_tensor(neighbors).cpu()
    }

    torch.save(checkpoint, CHECKPOINT_PATH)


def load_checkpoint():
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=True
    )

    model = SensorGRU()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    means = checkpoint["means"].numpy()
    stds = checkpoint["stds"].numpy()
    sensor_ids = checkpoint["sensor_ids"].numpy()
    neighbors = checkpoint["neighbors"].numpy()

    return model, means, stds, sensor_ids, neighbors

def predict_targets(model, values, availability, neighbors, times,
                    sensor_ids, targets, means, stds, batch_size=128):

    model.eval()

    id_to_index = {int(s): i for i, s in enumerate(sensor_ids)}
    time_to_index = {int(t): i for i, t in enumerate(times)}
    feature_to_index = {name: i for i, name in enumerate(MEASUREMENT_COLUMNS)}

    sequence_length = 40
    device = next(model.parameters()).device
    results = []

    with torch.inference_mode():
        for batch_start in range(0, len(targets), batch_size):
            batch = targets.iloc[batch_start:batch_start + batch_size]

            inputs = []
            target_ids = []
            target_steps = []
            target_features = []

            for row in batch.itertuples(index=False):
                sensor_index = id_to_index[int(row.nodeid)]
                time_index = time_to_index[int(row.time)]
                feature_index = feature_to_index[row.feature]

                if availability[time_index, sensor_index, feature_index]:
                    raise ValueError(f"Target {row.target_id} is visible")

                start = max(0, min(
                    time_index - sequence_length // 2,
                    len(times) - sequence_length
                ))
                end = start + sequence_length

                sensor_indices = neighbors[sensor_index]

                measurements = values[start:end, sensor_indices, :]
                flags = availability[start:end, sensor_indices, :]

                x = np.concatenate(
                    [measurements, flags.astype(np.float32)],
                    axis=2
                ).reshape(sequence_length, -1)

                inputs.append(x)
                target_ids.append(row.target_id)
                target_steps.append(time_index - start)
                target_features.append(feature_index)

            x = torch.from_numpy(
                np.stack(inputs).astype(np.float32)
            ).to(device)

            predictions = model(x)

            indices = torch.arange(len(batch), device=device)
            steps = torch.tensor(target_steps, device=device)
            features = torch.tensor(target_features, device=device)

            selected = predictions[indices, steps, features]
            selected = selected.cpu().numpy()

            for i, target_id in enumerate(target_ids):
                feature_index = target_features[i]

                prediction = (
                    selected[i] * stds[feature_index]
                    + means[feature_index]
                )

                results.append((target_id, float(prediction)))

    return pd.DataFrame(results, columns=["target_id", "prediction"])

def test_model():
    model, means, stds, sensor_ids, neighbors = load_checkpoint()

    test_df = pd.read_csv("data/masked/test_masked.csv")

    targets = pd.read_csv(
        "data/masked/test_answers.csv",
        usecols=["target_id", "time", "nodeid", "feature"]
    )

    values, availability, times, _ = build_sensor_arrays(
        test_df, sensor_ids
    )

    normalized = normalize_measurements(values, means, stds)

    predictions = predict_targets(
        model,
        normalized,
        availability,
        neighbors,
        times,
        sensor_ids,
        targets,
        means,
        stds
    )

    assert len(predictions) == len(targets)
    assert predictions["target_id"].is_unique
    assert np.isfinite(predictions["prediction"]).all()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    predictions.to_csv(PREDICTIONS_PATH, index=False)

    print(f"Saved {len(predictions):,} predictions to {PREDICTIONS_PATH}")

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["train", "test", "both"],
        default="both"
    )

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--samples-per-epoch", type=int, default=50000)

    args = parser.parse_args()

    if args.mode in ["train", "both"]:
        train_df = load_dataset("train")

        values, availability, times, sensor_ids = build_sensor_arrays(train_df)

        neighbors = find_nearest_neighbors(
            train_df, sensor_ids, NUM_NEIGHBORS
        )

        means, stds = fit_normalization(values)
        normalized = normalize_measurements(values, means, stds)

        dataset = SensorDataset(
            normalized, availability, neighbors
        )

        val_df = pd.read_csv("data/masked/val_masked.csv")
        val_answers = pd.read_csv("data/masked/val_answers.csv")

        val_values, val_availability, val_times, _ = build_sensor_arrays(
            val_df, sensor_ids
        )

        normalized_val = normalize_measurements(
            val_values, means, stds
        )

        val_data = (
            normalized_val,
            val_availability,
            val_times,
            val_answers
        )

        train_model(
            dataset,
            val_data,
            means,
            stds,
            sensor_ids,
            neighbors,
            epochs=args.epochs,
            batch_size=args.batch_size,
            samples_per_epoch=args.samples_per_epoch
        )

    if args.mode in ["test", "both"]:
        test_model()


if __name__ == "__main__":
    main()
