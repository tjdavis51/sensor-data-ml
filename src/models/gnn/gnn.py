# spatio-temporal graph neural network for reconstructing missing sensor
# measurements using both temporal history and the sensor graph

# general layout:
# graph: one node per sensor, edges to the k spatially nearest sensors
# node input: a window of 40 timesteps by 8 features
#   4 measurements (missing values set to zero) and 4 availability flags,
#   the same kind of input the GRU uses
# encoder: linear layer from 8 features to the hidden size
# 3 spatio-temporal blocks, each one:
#   temporal convolution: mixes information across nearby timesteps for each sensor
#   graph convolution: mixes information across neighboring sensors at each timestep
# linear output layer: hidden size to the 4 predicted measurements
# loss: masked MSE on artificially hidden values (same idea as the GRU)

# graph convolution (GCN, Kipf & Welling 2017):
# H' = A_hat * H * W
# H is the (nodes x features) matrix, W is a learned weight matrix
# A_hat = D^-1/2 (A + I) D^-1/2 is the normalized adjacency matrix
# multiplying by A_hat replaces each node's features with a weighted average
# of itself and its neighbors, so each layer lets information travel one hop
# with 3 blocks a sensor can use information from sensors up to 3 hops away

# the main difference from the GRU: the GRU reconstructs one sensor at a time from
# a fixed list of 4 neighbors, the GNN processes every sensor at once and
# learns how much to trust its neighbors through the graph

import os
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, RandomSampler

# directory containing the processed datasets
INPUT_DIR = "data/processed"
MASKED_DIR = "data/masked"

TIME_STEP = 30
MEASUREMENT_COLUMNS = ["temperature", "humidity", "light", "voltage"]

# graph and window settings
NUM_NEIGHBORS = 5       # edges from each sensor to its 5 nearest sensors
WINDOW_LENGTH = 40      # 40 timesteps = 20 minutes at 30 second intervals, same as the GRU
HIDDEN_SIZE = 64
NUM_BLOCKS = 3
KERNEL_SIZE = 5         # temporal convolution looks at 5 timesteps (2.5 minutes)

CHECKPOINT_DIR = "src/models/gnn/checkpoints"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pt")

OUTPUT_DIR = "src/models/gnn/results"
PREDICTIONS_PATH = os.path.join(OUTPUT_DIR, "predictions.csv")

# use a GPU when one is available
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")



# step 1: data arrays (same layout as the GRU: time x sensor x measurement)

def load_dataset(path):

    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} observations from {path}")
    return df


def build_sensor_arrays(df, sensor_ids=None):

    # use the dataset sensor ids unless a specific ordering is provided
    if sensor_ids is None:
        sensor_ids = np.sort(df["nodeid"].unique())

    # verify that the timestamps are aligned to 30 second intervals
    assert (df["time"] % TIME_STEP == 0).all()

    # complete timeline from the first to the last timestamp
    times = np.arange(
        int(df["time"].min()),
        int(df["time"].max()) + TIME_STEP,
        TIME_STEP
    )

    # align measurements with every time and sensor combination
    # missing combinations become rows of NaN
    full_index = pd.MultiIndex.from_product(
        [times, sensor_ids], names=["time", "nodeid"]
    )
    measurements = (
        df.set_index(["time", "nodeid"])[MEASUREMENT_COLUMNS]
        .reindex(full_index)
    )

    # reshape into time x sensor x measurement
    values = measurements.to_numpy(dtype=np.float32).reshape(
        len(times), len(sensor_ids), len(MEASUREMENT_COLUMNS)
    )

    # true where a measurement exists
    availability = ~np.isnan(values)

    print(f"Sensor array shape: {values.shape}")
    return values, availability, times, sensor_ids


def fit_normalization(values):

    # training statistics for each measurement
    means = np.nanmean(values, axis=(0, 1))
    stds = np.nanstd(values, axis=(0, 1))
    stds[stds == 0] = 1.0
    return means.astype(np.float32), stds.astype(np.float32)


def normalize_measurements(values, means, stds):

    # z-score each measurement and replace missing values with zero
    # the availability flags tell the model which zeros are real
    normalized = (values - means) / stds
    return np.nan_to_num(normalized, nan=0.0).astype(np.float32)


# step 2: the sensor graph

def get_positions(df, sensor_ids):

    # one (x, y) row per sensor in the same order as sensor_ids
    positions = (
        df.drop_duplicates("nodeid")
        .set_index("nodeid")
        .loc[sensor_ids, ["x", "y"]]
        .to_numpy(dtype=np.float32)
    )
    return positions


def build_adjacency(positions, num_neighbors=NUM_NEIGHBORS):

    num_sensors = len(positions)

    # pairwise euclidean distances between sensors
    differences = positions[:, None, :] - positions[None, :, :]
    distances = np.sqrt((differences ** 2).sum(axis=2))

    # connect each sensor to its k nearest sensors (index 0 is itself, so skip it)
    nearest = np.argsort(distances, axis=1)[:, 1:num_neighbors + 1]

    # closer sensors get stronger edges: weight = exp(-(d / sigma)^2)
    # sigma is the median neighbor distance, so a typical neighbor gets a weight
    # of about 0.37 and the farthest neighbors get weights close to 0
    sigma = np.median(np.take_along_axis(distances, nearest, axis=1))

    adjacency = np.zeros((num_sensors, num_sensors), dtype=np.float32)
    for i in range(num_sensors):
        for j in nearest[i]:
            weight = np.exp(-(distances[i, j] / sigma) ** 2)
            # make the graph undirected: if i is near j then j is near i
            adjacency[i, j] = weight
            adjacency[j, i] = weight

    # add self loops so each node keeps its own information
    adjacency = adjacency + np.eye(num_sensors, dtype=np.float32)

    # symmetric normalization: D^-1/2 (A + I) D^-1/2
    degree = adjacency.sum(axis=1)
    d_inv_sqrt = 1.0 / np.sqrt(degree)
    normalized = d_inv_sqrt[:, None] * adjacency * d_inv_sqrt[None, :]

    edges = int((adjacency > 0).sum() - num_sensors) // 2
    print(f"Graph: {num_sensors} sensors, {edges} undirected edges")

    return normalized.astype(np.float32)


# step 3: training windows and input features

def build_window(window_values, window_flags):

    # build the 8 model input features for one window:
    # 4 measurements (missing values are already zero) and 4 availability flags
    # window_values / window_flags: (window, sensor, measurement)
    # returns an array of shape (sensor, window, 8)

    x = np.concatenate([window_values, window_flags.astype(np.float32)], axis=-1)

    # (window, sensor, 8) -> (sensor, window, 8)
    return x.transpose(1, 0, 2).astype(np.float32)


class GraphWindowDataset(Dataset):

    # each sample is one time window for every sensor at once
    # x:    sensors x window x 8 (input features from build_window)
    # y:    sensors x window x 4 (original normalized measurements)
    # mask: sensors x window x 4 (true where a value was hidden for training)

    def __init__(self, values, availability, window_length=WINDOW_LENGTH, mask_probability=0.1):

        self.values = values
        self.availability = availability
        self.window_length = window_length
        self.mask_probability = mask_probability

    def __len__(self):
        return len(self.values) - self.window_length + 1

    def __getitem__(self, start):

        end = start + self.window_length

        window_values = self.values[start:end].copy()
        window_flags = self.availability[start:end].copy()

        # time x sensor x feature -> sensor x time x feature
        y = window_values.transpose(1, 0, 2).copy()

        # randomly hide 10 percent of the available measurements
        mask = (np.random.random(window_values.shape) < self.mask_probability) & window_flags
        window_values[mask] = 0.0
        window_flags[mask] = False

        x = build_window(window_values, window_flags)

        return (
            torch.from_numpy(x),
            torch.from_numpy(y),
            torch.from_numpy(mask.transpose(1, 0, 2).copy()),
        )


# step 4: the model

class GraphConv(nn.Module):

    # one graph convolution: H' = A_hat * H * W
    # works on batches of shape (batch, sensors, time, features)

    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, h, adjacency):

        # transform each node's features
        h = self.linear(h)

        # average over neighbors: for every batch and timestep,
        # new node n = sum over m of adjacency[n, m] * h[m]
        return torch.einsum("nm,bmtf->bntf", adjacency, h)


class SpatioTemporalBlock(nn.Module):

    # temporal convolution, then graph convolution, each with a residual connection

    def __init__(self, hidden_size, kernel_size=KERNEL_SIZE, dropout=0.1):
        super().__init__()

        # padding keeps the window length the same
        self.temporal = nn.Conv1d(
            hidden_size, hidden_size, kernel_size, padding=kernel_size // 2
        )
        self.spatial = GraphConv(hidden_size, hidden_size)

        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, adjacency):

        batch, sensors, time, features = h.shape

        # temporal convolution: Conv1d expects (items, channels, time)
        # so treat every (batch, sensor) pair as a separate sequence
        t = h.reshape(batch * sensors, time, features).transpose(1, 2)
        t = self.temporal(t).transpose(1, 2).reshape(batch, sensors, time, features)
        h = self.norm1(h + self.dropout(self.activation(t)))

        # graph convolution across sensors at each timestep
        s = self.spatial(h, adjacency)
        h = self.norm2(h + self.dropout(self.activation(s)))

        return h


class SensorGNN(nn.Module):

    def __init__(self, adjacency, hidden_size=HIDDEN_SIZE, num_blocks=NUM_BLOCKS):
        super().__init__()

        # store the graph with the model so it moves to the device and is saved
        self.register_buffer("adjacency", torch.as_tensor(adjacency))

        # learned embedding for each sensor so the model can learn
        # sensor specific behavior (for example a sensor that always reads warm)
        num_sensors = adjacency.shape[0]
        self.sensor_embedding = nn.Embedding(num_sensors, hidden_size)

        # 8 input features: 4 measurements and 4 availability flags
        self.input = nn.Linear(2 * len(MEASUREMENT_COLUMNS), hidden_size)
        self.blocks = nn.ModuleList(
            [SpatioTemporalBlock(hidden_size) for _ in range(num_blocks)]
        )
        self.output = nn.Linear(hidden_size, len(MEASUREMENT_COLUMNS))

    def forward(self, x):

        # x: (batch, sensors, time, 8)
        h = self.input(x)

        # add the sensor embedding to every timestep
        h = h + self.sensor_embedding.weight[None, :, None, :]

        for block in self.blocks:
            h = block(h, self.adjacency)

        # predictions: (batch, sensors, time, 4)
        return self.output(h)


def masked_mse_loss(predictions, targets, mask):

    # mse only on the artificially hidden values
    return torch.mean((predictions[mask] - targets[mask]) ** 2)


# step 5: predicting the fixed validation and test targets

def predict_targets(model, values, availability, times, sensor_ids, targets,
                    means, stds, window_length=WINDOW_LENGTH, batch_size=64):

    # predict every hidden target in a targets dataframe
    # (columns: target_id, time, nodeid, feature)
    # one window is run per unique target timestamp, centered on that timestamp,
    # which predicts every sensor and measurement at that time in one pass

    model.eval()
    device = next(model.parameters()).device

    id_to_index = {int(s): i for i, s in enumerate(sensor_ids)}
    time_to_index = {int(t): i for i, t in enumerate(times)}
    feature_to_index = {name: i for i, name in enumerate(MEASUREMENT_COLUMNS)}

    time_idx = targets["time"].astype(int).map(time_to_index).to_numpy()
    sensor_idx = targets["nodeid"].astype(int).map(id_to_index).to_numpy()
    feature_idx = targets["feature"].map(feature_to_index).to_numpy()

    # verify that no answer is visible to the model
    if availability[time_idx, sensor_idx, feature_idx].any():
        raise ValueError("Some targets are visible in the masked input")

    # one centered window per unique target timestamp
    unique_times = np.unique(time_idx)
    starts = np.clip(unique_times - window_length // 2, 0, len(times) - window_length)

    # normalized prediction at every (target time, sensor, feature)
    center_predictions = np.zeros(
        (len(unique_times), len(sensor_ids), len(MEASUREMENT_COLUMNS)), dtype=np.float32
    )

    with torch.inference_mode():
        for batch_start in range(0, len(unique_times), batch_size):

            batch_starts = starts[batch_start:batch_start + batch_size]
            batch_times = unique_times[batch_start:batch_start + batch_size]

            # build the same 8 input features used in training: (batch, sensor, time, 8)
            x = np.stack([
                build_window(values[s:s + window_length], availability[s:s + window_length])
                for s in batch_starts
            ])
            x = torch.from_numpy(x).to(device)

            predictions = model(x).cpu().numpy()

            # pick the timestep inside each window that matches the target time
            steps = batch_times - batch_starts
            center_predictions[batch_start:batch_start + len(batch_times)] = (
                predictions[np.arange(len(steps)), :, steps, :]
            )

    # look up each target and convert back to original units
    row = np.searchsorted(unique_times, time_idx)
    normalized = center_predictions[row, sensor_idx, feature_idx]
    predictions = normalized * stds[feature_idx] + means[feature_idx]

    return pd.DataFrame({
        "target_id": targets["target_id"].to_numpy(),
        "prediction": predictions.astype(np.float64),
    })


def validation_nrmse(predictions, answers, training_std):

    # same scoring as evaluation.py: RMSE / training std, averaged over measurements
    merged = answers.merge(predictions, on="target_id")
    scores = {}
    for column in MEASUREMENT_COLUMNS:
        rows = merged[merged["feature"] == column]
        rmse = np.sqrt(np.mean((rows["prediction"] - rows["true_value"]) ** 2))
        scores[column] = rmse / training_std[column]
    return float(np.mean(list(scores.values()))), scores


# step 6: training and testing

def save_checkpoint(model, means, stds, sensor_ids, adjacency, window_length):

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "means": torch.as_tensor(means),
        "stds": torch.as_tensor(stds),
        "sensor_ids": torch.as_tensor(sensor_ids),
        "adjacency": torch.as_tensor(adjacency),
        # needed to build the same windows at test time
        "window_length": window_length,
    }, CHECKPOINT_PATH)


def load_checkpoint():

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)

    adjacency = checkpoint["adjacency"].numpy()
    model = SensorGNN(adjacency)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE).eval()

    return (
        model,
        checkpoint["means"].numpy(),
        checkpoint["stds"].numpy(),
        checkpoint["sensor_ids"].numpy(),
        checkpoint["window_length"],
    )


def prepare_masked_split(name, sensor_ids, means, stds):

    # load a fixed masked split and its target list, using the training sensor order
    masked_df = load_dataset(os.path.join(MASKED_DIR, f"{name}_masked.csv"))
    answers = pd.read_csv(os.path.join(MASKED_DIR, f"{name}_answers.csv"))

    # keep only sensors the model knows about
    masked_df = masked_df[masked_df["nodeid"].isin(sensor_ids)]

    values, availability, times, _ = build_sensor_arrays(masked_df, sensor_ids)
    normalized = normalize_measurements(values, means, stds)

    return normalized, availability, times, answers


def train(epochs, batch_size, samples_per_epoch, learning_rate, seed, window_length):

    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"Using device: {DEVICE}")

    # training data (original values, hidden dynamically each batch like the GRU)
    train_df = load_dataset(os.path.join(INPUT_DIR, "train.csv"))
    values, availability, times, sensor_ids = build_sensor_arrays(train_df)

    means, stds = fit_normalization(values)
    normalized = normalize_measurements(values, means, stds)

    # build the sensor graph from the positions
    positions = get_positions(train_df, sensor_ids)
    adjacency = build_adjacency(positions)

    # training std in original units, used for NRMSE exactly like evaluation.py
    training_std = train_df[MEASUREMENT_COLUMNS].std()

    # fixed validation masks shared by every model
    val_values, val_availability, val_times, val_answers = prepare_masked_split(
        "val", sensor_ids, means, stds
    )
    val_targets = val_answers[["target_id", "time", "nodeid", "feature"]]

    dataset = GraphWindowDataset(normalized, availability, window_length=window_length)
    sampler = RandomSampler(
        dataset, replacement=False, num_samples=min(samples_per_epoch, len(dataset))
    )
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    model = SensorGNN(adjacency).to(DEVICE)
    print(f"Window: {window_length} steps")
    num_parameters = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_parameters:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_score = float("inf")

    for epoch in range(epochs):
        model.train()
        total_loss, total_batches = 0.0, 0

        for x, y, mask in loader:
            x, y, mask = x.to(DEVICE), y.to(DEVICE), mask.to(DEVICE)

            if not mask.any():
                continue

            optimizer.zero_grad()
            loss = masked_mse_loss(model(x), y, mask)
            loss.backward()

            # keep updates stable
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_batches += 1

        scheduler.step()

        # score the fixed validation targets with the shared metric
        val_predictions = predict_targets(
            model, val_values, val_availability, val_times, sensor_ids,
            val_targets, means, stds, window_length=window_length
        )
        score, per_feature = validation_nrmse(val_predictions, val_answers, training_std)

        details = "  ".join(f"{k[:4]} {v:.3f}" for k, v in per_feature.items())
        print(
            f"Epoch {epoch + 1}/{epochs} - train MSE {total_loss / max(total_batches, 1):.4f}"
            f" - val NRMSE {score:.4f} ({details})"
        )

        # keep the checkpoint with the best validation score
        if score < best_score:
            best_score = score
            save_checkpoint(model, means, stds, sensor_ids, adjacency, window_length)
            print(f"  saved best model (val NRMSE {best_score:.4f})")


def test():

    model, means, stds, sensor_ids, window_length = load_checkpoint()

    values, availability, times, answers = prepare_masked_split(
        "test", sensor_ids, means, stds
    )

    # only pass the target locations, never the true values
    targets = answers[["target_id", "time", "nodeid", "feature"]]

    predictions = predict_targets(
        model, values, availability, times, sensor_ids, targets, means, stds,
        window_length=window_length
    )

    # verify that every target received one finite prediction
    assert len(predictions) == len(targets)
    assert predictions["target_id"].is_unique
    assert np.isfinite(predictions["prediction"]).all()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    predictions.to_csv(PREDICTIONS_PATH, index=False)
    print(f"Saved {len(predictions):,} predictions to {PREDICTIONS_PATH}")


def main():

    # defaults match the GRU's main settings: 20 minute windows,
    # batch size 64 and 5 epochs
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "test", "both"], default="both")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--samples-per-epoch", type=int, default=4000)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window-length", type=int, default=WINDOW_LENGTH)
    args = parser.parse_args()

    if args.mode in ["train", "both"]:
        train(args.epochs, args.batch_size, args.samples_per_epoch,
              args.learning_rate, args.seed, args.window_length)

    if args.mode in ["test", "both"]:
        test()


if __name__ == "__main__":
    main()