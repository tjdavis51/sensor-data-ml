"""Feed-forward sensor reconstruction model and command-line entry point."""

import sys
from pathlib import Path

from torch import nn

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.utils.masking import MEASUREMENT_COLUMNS


class SensorMLP(nn.Module):
    """Predict the four target-node measurements from a vector of available inputs."""

    def __init__(self, input_size: int, hidden_sizes=(64,)):
        super().__init__()
        if not hidden_sizes or min(hidden_sizes) < 1:
            raise ValueError("Hidden layer sizes must be positive")
        layers = []
        previous_size = input_size
        for size in hidden_sizes:
            layers.extend([nn.Linear(previous_size, size), nn.ReLU()])
            previous_size = size
        layers.append(nn.Linear(previous_size, len(MEASUREMENT_COLUMNS)))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.network(inputs)


if __name__ == "__main__":
    from src.models.neural_network.training import main

    main()
