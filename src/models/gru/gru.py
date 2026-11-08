# bidirectional GRU for reconstructing missing sensor
# measurments using both temportal history and nearby sensors

# general layout:
# target sensor and the 4 nearest neighbors: looks at temp, humididy, light, voltage
# sequence construction: 40 timesteps by 40 input features
# bidirectional GRU: layer 1 with 128 hidden units per direction
# bidirectional GRU: layer 2 with 128 hidden units per direction
# linear out layer: 256 to 4 predicted measurements
# then get the masked reconstruction loss on the hidden values

# parameters:
# reconstruct 1 target sensor at a time
# using the 4 spacially nearest neighbors
# sequence length can be 40 timesteps, which works out to 20 minutes at 30 second intervals
# 4 measruements, temp, humididty, light, voltage
# possible masks: 4 per sensor
# 40 input feartures (5 sensors by 8 features)
# I'll use 2 GRU layers to start with
# and a hidden size of 128
# I'll set it to be bidirectional to get past and future context
# with 4 output features
# and the loss function will be the masked MSE, trained only on hidden targets

# GRU:
# recurrent neural network that uses gates to decide what information to retain from the previous timesteps
# Update Gate: z_t = sigma(W_z * x_t + U_z * h_t-1 + b_z)
# the update gate determines how much information from the previous hidden state should be preserved
# if z_t is close to 1 then the GRU retains more of its previous state under the convention below
# if is clost to 0 then it incorporates more of the nealy computed candidate state
# Reset Gate: r_t = sigma(W_r * x_t + U_r * h_t-1 + b_r)
# the reset gate controls how much previous information influences the candidate hidden state
# Candidate hidden state: h_t = tanh(W_h * x_t + U_h(r_t element wise h_t-1) + b_h)
# Final Hidden State: h_t = z_t element wise h_t-1 + (1 - z_t) element wise h-til_t
# this combines the previous hidden state with the candidate hidden state
# the main benefit is that the GRU can learn relationships across lots of timesteps without
# suffering from the vanishing gradient problems from most recurrent networks

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler

import pandas as pd
import os
import numpy as np
import argparse
# directory containing the processed datasets
INPUT_DIR = "data/processed"

TIME_STEP = 30
MEASUREMENT_COLUMNS = ["temperature", "humidity", "light", "voltage"]
NUM_NEIGHBORS = 4

CHECKPOINT_DIR = "src/models/gru/checkpoints"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pt")

OUTPUT_DIR = "src/models/gru/results"
PREDICTIONS_PATH = os.path.join(OUTPUT_DIR, "predictions.csv")

DEVICE = torch.device("cpu") # Metal setting are "mps"

print(f"Using device: {DEVICE}")

# first phase will be to build the data pipeline to load the synchronized data,
# construct the sensor time arrays, find nearest neighbors,
# normalize measurements, and produce the tensors with shape = (batch, sequence, features)

# function to load a processed dataset
def load_dataset(name):

    # create the path to the dataset
    path = os.path.join(INPUT_DIR, f"{name}.csv")

    # read the csv file into a pandas dataframe
    df = pd.read_csv(path)

    # print the number of observations loaded
    print(f"Loaded {len(df):,} {name} observations")

    # return the dataframe
    return df

def build_sensor_arrays(df, sensor_ids=None):

    # use the dataset sensor ids unless a specific ordering is provided
    if sensor_ids is None:
        sensor_ids = np.sort(df["nodeid"].unique())

    # create a complete timeline at 30 second intervals
    # so get the start and end times from the data
    start_time = int(df["time"].min())
    end_time = int(df["time"].max())

    # populate the numpy array of 30 second intevals from beginning to end
    times = np.arange(
        start_time,
        # make sure to add 1 more time step to the end because it might not line up with the previous interval
        end_time + TIME_STEP,
        TIME_STEP
    )

    # very important to check
    # verify that the timestamps are aligned to 30 second intervals
    assert (df["time"] % TIME_STEP == 0).all()

    # get a table indexed by the existing time and sensor id pairs
    indexed_df = df.set_index(["time", "nodeid"])

    # create every possible time and sensor combination
    full_index = pd.MultiIndex.from_product(
        [times, sensor_ids],
        names=["time", "nodeid"]
    )

    # align measurements with the complete time sensor grid
    # first select the measurement column
    measurements = indexed_df[MEASUREMENT_COLUMNS]
    # then reorganize the observations to match the oder of full index
    # if a requested time sensor combination doesn't exist then insert a row of missing values
    measurements = measurements.reindex(full_index)

    # convert the pandas dataframe into a numpy array
    # removing the pandas index and the column labels
    values = measurements.to_numpy(dtype=np.float32)

    # the shape should look something like this:
    # array([[measurements at time 30 and first sensor], [...], ...])

    # reshape into a tensor with shape: time by sensor by measurement
    values = values.reshape(
        len(times),
        len(sensor_ids),
        len(MEASUREMENT_COLUMNS)
    )

    # create availability indicators, true if a measurement is available and false if not
    availability = ~np.isnan(values)

    # print the array dimensions
    print(f"Sensor array shape: {values.shape}")

    # return the arrays and their corresponding identifiers
    return values, availability, times, sensor_ids

def squared_eu_distance(loc1, loc2):
    return (loc1[0] - loc2[0]) ** 2 + (loc1[1] - loc2[1]) ** 2

def find_nearest_neighbors(df, sensor_ids, num_neighbors):
    # create a dictionary to store each sensor and its neighbors
    sensors = {}

    # loop through each sensor id
    for sensor_id in sensor_ids:

        # get the row containing this sensor's position
        sensor = df[df["nodeid"] == sensor_id].iloc[0]

        # get the sensor coordinates
        position = (sensor["x"], sensor["y"])

        # initialize the record with empty neighbors
        sensors[int(sensor_id)] = {
            "position": position,
            "neighbors": [None] * num_neighbors,
            "distances": [float("inf")] * num_neighbors
        }

    # loop through each pair of sensors
    for key_id in sensors:
        for comparison_id in sensors:

            # skip comparing a sensor with itself
            if key_id == comparison_id:
                continue

            # calculate squared euclidean distance
            dist = squared_eu_distance(
                sensors[key_id]["position"],
                sensors[comparison_id]["position"]
            )

            # insert the new neighbor if it is closer
            for i in range(num_neighbors):
                if dist < sensors[key_id]["distances"][i]:

                    sensors[key_id]["distances"].insert(i, dist)
                    sensors[key_id]["neighbors"].insert(i, comparison_id)

                    sensors[key_id]["distances"].pop()
                    sensors[key_id]["neighbors"].pop()

                    break

    # map sensor ids to array indices
    id_to_index = {
        int(sensor_id): i
        for i, sensor_id in enumerate(sensor_ids)
    }

    # prepare the return array
    neighbors = []

    for sensor_id in sensor_ids:
        sensor_id = int(sensor_id)

        # include the target sensor first
        id_with_neighbors = [sensor_id] + sensors[sensor_id]["neighbors"]

        # convert ids to array indices
        index_with_neighbors = [
            id_to_index[neighbor_id]
            for neighbor_id in id_with_neighbors
        ]

        neighbors.append(index_with_neighbors)

    return np.array(neighbors, dtype=np.int32)

def fit_normalization(values):
    # calculate training statistics for each measurement
    means = np.nanmean(values, axis=(0, 1))
    stds = np.nanstd(values, axis=(0, 1))

    # prevent division by zero
    stds[stds == 0] = 1.0

    return means, stds

def normalize_measurements(values, means, stds):
    # normalize each measurement using training statistics
    normalized = (values - means) / stds

    # replace missing measurements with zero
    normalized = np.nan_to_num(normalized, nan=0.0)

    return normalized

# def build_sequences(values, availability, neighbors, sequence_length=40):
#     # select a target sensor and its nearest neighbors
#     # extract 40 consecutive timesteps from the normalized measurments
#     # extract the corresponding availability indicators
#     # combine the measurements and availability indicators into eight features per sensor
#     # flatten the five sensors into 40 input features per timestep
#     pass

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
        # number of possible windows for each sensor
        windows_per_sensor = self.num_times - self.sequence_length + 1

        return windows_per_sensor * self.num_sensors

    def __getitem__(self, index):
        # determine the target sensor and starting timestep
        windows_per_sensor = self.num_times - self.sequence_length + 1

        sensor_index = index // windows_per_sensor
        start = index % windows_per_sensor
        end = start + self.sequence_length

        # get the target sensor and its neighbors
        sensor_indices = self.neighbors[sensor_index]

        # extract measurements and availability flags
        measurements = self.values[start:end, sensor_indices, :].copy()
        availability = self.availability[start:end, sensor_indices, :].copy()

        # save the original target measurements
        y = measurements[:, 0, :].copy()

        # randomly hide 10 percent of available target measurements
        mask = (
            np.random.random(y.shape) < self.mask_probability
        ) & availability[:, 0, :]

        # hide selected values from the input
        measurements[:, 0, :][mask] = 0.0
        availability[:, 0, :][mask] = False

        # combine measurements and availability
        x = torch.cat([
            torch.tensor(measurements, dtype=torch.float32),
            torch.tensor(availability, dtype=torch.float32)
        ], dim=2)

        # flatten sensor and feature dimensions
        x = x.reshape(self.sequence_length, -1)

        return (
            x,
            torch.tensor(y, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.bool)
        )


# implement the bidirectional GRU
# verify that it produces the expected output shape
class SensorGRU(nn.Module):

    def __init__(self):

        super().__init__()

        # process sensor sequences in both directions
        self.gru = nn.GRU(
            input_size=40,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        # predict the four measurements of the target sensor
        self.output = nn.Linear(256, 4)

    def forward(self, x):

        # process the input sequence
        hidden_states, _ = self.gru(x)

        # predict the target sensor measurements
        predictions = self.output(hidden_states)

        return predictions

# implement training and validation with dynamic masksing, masked MSE loss,
# Adam optimization, and checkpoint selection using validation performance

def masked_mse_loss(predictions, targets, mask):
    # verify that the batch contains hidden targets
    if not mask.any().item():
        raise ValueError("No masked targets in batch")

    # calculate mse only on artificially hidden values
    return torch.mean((predictions[mask] - targets[mask]) ** 2)

def train_model(dataset, val_data, means, stds, sensor_ids, neighbors,
                epochs=5, batch_size=64, learning_rate=0.001, samples_per_epoch=50000):
    
    # create the gru
    model = SensorGRU().to(DEVICE)

    # configure the optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    # sample a subset of the available sequences each epoch
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

    # begin training
    for epoch in range(epochs):
        model.train()

        total_loss = 0.0
        total_batches = 0

        for x, y, mask in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            mask = mask.to(DEVICE)
            # skip batches without masked targets
            if not mask.any():
                continue

            # clear previous gradients
            optimizer.zero_grad()

            # predict missing measurements
            predictions = model(x)

            # calculate reconstruction loss
            loss = masked_mse_loss(predictions, y, mask)

            # compute gradients
            loss.backward()

            # update network weights
            optimizer.step()

            total_loss += loss.item()
            total_batches += 1

        average_loss = total_loss / max(total_batches, 1)

        print(f"Epoch {epoch + 1}/{epochs} - Training MSE: {average_loss:.4f}")


        # calculate validation loss after each epoch
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

        # save the model if validation performance improved
        if np.isfinite(val_loss) and val_loss < best_val_loss:
            best_val_loss = val_loss

            save_checkpoint(
                model, means, stds, sensor_ids, neighbors
            )

            print(f"Saved best model with validation MSE: {best_val_loss:.4f}")


    return model

# implement the prediction and evaluation by first
# generating predictions for the fixed validation and test masks
# save the predictions, and prepare for evaluation/benchmarking


def validate_model(model, values, availability, neighbors, times,
                   sensor_ids, answers, means, stds, batch_size=128):

    # switch the model to evaluation mode
    model.eval()

    # map actual sensor ids and timestamps to array indices
    id_to_index = {int(sensor_id): i for i, sensor_id in enumerate(sensor_ids)}
    time_to_index = {int(time): i for i, time in enumerate(times)}
    feature_to_index = {name: i for i, name in enumerate(MEASUREMENT_COLUMNS)}

    sequence_length = 40
    device = next(model.parameters()).device

    total_squared_error = 0.0
    total_predictions = 0

    # process validation targets in batches
    with torch.inference_mode():
        for batch_start in range(0, len(answers), batch_size):
            batch = answers.iloc[batch_start:batch_start + batch_size]

            inputs = []
            target_values = []
            target_steps = []
            target_features = []

            for row in batch.itertuples(index=False):
                # identify the hidden measurement
                sensor_index = id_to_index[int(row.nodeid)]
                time_index = time_to_index[int(row.time)]
                feature_index = feature_to_index[row.feature]

                # verify that the correct answer is hidden from the input
                if availability[time_index, sensor_index, feature_index]:
                    raise ValueError(
                        f"Validation target is visible: "
                        f"time={row.time}, "
                        f"nodeid={row.nodeid}, "
                        f"feature={row.feature}"
                    )

                # choose a window around the target timestamp
                start = max(0, min(
                    time_index - sequence_length // 2,
                    len(times) - sequence_length
                ))
                end = start + sequence_length

                # get the target sensor and its four neighbors
                sensor_indices = neighbors[sensor_index]

                # extract the normalized measurements and availability flags
                measurements = values[start:end, sensor_indices, :]
                flags = availability[start:end, sensor_indices, :]

                # combine measurements and flags into 40 input features
                x = np.concatenate(
                    [measurements, flags.astype(np.float32)],
                    axis=2
                ).reshape(sequence_length, -1)

                inputs.append(x)

                # normalize the correct answer using training statistics
                target = (row.true_value - means[feature_index]) / stds[feature_index]

                target_values.append(target)
                target_steps.append(time_index - start)
                target_features.append(feature_index)

            # run one batch through the gru
            x = torch.from_numpy(np.stack(inputs)).to(device)
            predictions = model(x)

            # select the prediction for each hidden target
            indices = torch.arange(len(batch), device=device)
            steps = torch.tensor(target_steps, device=device)
            features = torch.tensor(target_features, device=device)

            selected_predictions = predictions[indices, steps, features]

            # calculate squared error in normalized units
            targets = torch.tensor(target_values, dtype=torch.float32, device=device)
            squared_errors = (selected_predictions - targets) ** 2

            total_squared_error += squared_errors.sum().item()
            total_predictions += len(batch)

    return total_squared_error / total_predictions


def save_checkpoint(model, means, stds, sensor_ids, neighbors):
    # create the checkpoint directory
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # save weights and preprocessing information
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "means": torch.as_tensor(means).cpu(),
        "stds": torch.as_tensor(stds).cpu(),
        "sensor_ids": torch.as_tensor(sensor_ids).cpu(),
        "neighbors": torch.as_tensor(neighbors).cpu()
    }

    torch.save(checkpoint, CHECKPOINT_PATH)


def load_checkpoint():
    # load the saved checkpoint
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=True
    )

    # recreate the network and restore its weights
    model = SensorGRU()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # restore preprocessing information
    means = checkpoint["means"].numpy()
    stds = checkpoint["stds"].numpy()
    sensor_ids = checkpoint["sensor_ids"].numpy()
    neighbors = checkpoint["neighbors"].numpy()

    return model, means, stds, sensor_ids, neighbors

def predict_targets(model, values, availability, neighbors, times,
                    sensor_ids, targets, means, stds, batch_size=128):

    # switch to evaluation mode
    model.eval()

    # create lookup tables for array indices
    id_to_index = {int(s): i for i, s in enumerate(sensor_ids)}
    time_to_index = {int(t): i for i, t in enumerate(times)}
    feature_to_index = {name: i for i, name in enumerate(MEASUREMENT_COLUMNS)}

    sequence_length = 40
    device = next(model.parameters()).device
    results = []

    # generate predictions in batches
    with torch.inference_mode():
        for batch_start in range(0, len(targets), batch_size):
            batch = targets.iloc[batch_start:batch_start + batch_size]

            inputs = []
            target_ids = []
            target_steps = []
            target_features = []

            for row in batch.itertuples(index=False):
                # locate the hidden target
                sensor_index = id_to_index[int(row.nodeid)]
                time_index = time_to_index[int(row.time)]
                feature_index = feature_to_index[row.feature]

                # verify that the answer is not visible
                if availability[time_index, sensor_index, feature_index]:
                    raise ValueError(f"Target {row.target_id} is visible")

                # select the same centered window used for validation
                start = max(0, min(
                    time_index - sequence_length // 2,
                    len(times) - sequence_length
                ))
                end = start + sequence_length

                # select target sensor and nearest neighbors
                sensor_indices = neighbors[sensor_index]

                measurements = values[start:end, sensor_indices, :]
                flags = availability[start:end, sensor_indices, :]

                # combine measurements with availability indicators
                x = np.concatenate(
                    [measurements, flags.astype(np.float32)],
                    axis=2
                ).reshape(sequence_length, -1)

                inputs.append(x)
                target_ids.append(row.target_id)
                target_steps.append(time_index - start)
                target_features.append(feature_index)

            # run the gru on the batch
            x = torch.from_numpy(
                np.stack(inputs).astype(np.float32)
            ).to(device)

            predictions = model(x)

            # select the relevant output for each target
            indices = torch.arange(len(batch), device=device)
            steps = torch.tensor(target_steps, device=device)
            features = torch.tensor(target_features, device=device)

            selected = predictions[indices, steps, features]
            selected = selected.cpu().numpy()

            # convert predictions back to original measurement units
            for i, target_id in enumerate(target_ids):
                feature_index = target_features[i]

                prediction = (
                    selected[i] * stds[feature_index]
                    + means[feature_index]
                )

                results.append((target_id, float(prediction)))

    # return the predictions in benchmark format
    return pd.DataFrame(results, columns=["target_id", "prediction"])

def test_model():
    # load the trained model and preprocessing information
    model, means, stds, sensor_ids, neighbors = load_checkpoint()

    # load the masked test measurements
    test_df = pd.read_csv("data/masked/test_masked.csv")

    # load only the metadata identifying hidden test targets
    targets = pd.read_csv(
        "data/masked/test_answers.csv",
        usecols=["target_id", "time", "nodeid", "feature"]
    )

    # organize test measurements using training sensor ordering
    values, availability, times, _ = build_sensor_arrays(
        test_df, sensor_ids
    )

    # normalize using saved training statistics
    normalized = normalize_measurements(values, means, stds)

    # predict the hidden measurements
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

    # verify that every target received one prediction
    assert len(predictions) == len(targets)
    assert predictions["target_id"].is_unique
    assert np.isfinite(predictions["prediction"]).all()

    # save the predictions
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

    # train and save the best gru
    if args.mode in ["train", "both"]:
        train_df = load_dataset("train")

        # prepare training arrays
        values, availability, times, sensor_ids = build_sensor_arrays(train_df)

        # find neighbors
        neighbors = find_nearest_neighbors(
            train_df, sensor_ids, NUM_NEIGHBORS
        )

        # fit normalization using training data only
        means, stds = fit_normalization(values)
        normalized = normalize_measurements(values, means, stds)

        # prepare training dataset
        dataset = SensorDataset(
            normalized, availability, neighbors
        )

        # load fixed validation data
        val_df = pd.read_csv("data/masked/val_masked.csv")
        val_answers = pd.read_csv("data/masked/val_answers.csv")

        # use training sensor ordering
        val_values, val_availability, val_times, _ = build_sensor_arrays(
            val_df, sensor_ids
        )

        # normalize validation measurements
        normalized_val = normalize_measurements(
            val_values, means, stds
        )

        val_data = (
            normalized_val,
            val_availability,
            val_times,
            val_answers
        )

        # train and select the best checkpoint using validation
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

    # load checkpoint and generate test predictions
    if args.mode in ["test", "both"]:
        test_model()


if __name__ == "__main__":
    main()

