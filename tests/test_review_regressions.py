"""Regression checks for anomaly metrics, target masking, and missing sensor context."""

import contextlib
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from anomaly_detection.anomaly_detection import predict_targets
from anomaly_detection.evaluate_anomaly_detection import evaluate_labels, load_detections
from src.models.linear_regression.linear_regression import nearest_observed

ROOT = Path(__file__).resolve().parents[1]


class TemporalNeighborsTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({
            "time": [0, 30, 60, 0, 30, 60],
            "nodeid": [1, 1, 1, 2, 2, 2],
            "temperature": [np.nan, np.nan, np.nan, 10.0, np.nan, 20.0],
        })

    def test_all_missing_sensor_does_not_borrow_another_sensors_readings(self):
        targets = pd.DataFrame({"time": [30], "nodeid": [1]})
        result = nearest_observed(self.frame, "temperature", targets, 20)
        self.assertTrue(all(np.isnan(values).all() for values in result))

    def test_visible_neighbors_keep_their_values_and_distances(self):
        targets = pd.DataFrame({"time": [30], "nodeid": [2]})
        result = nearest_observed(self.frame, "temperature", targets, 20)
        np.testing.assert_allclose(np.array(result).ravel(), [10, 20, 1, 1])

    def test_all_missing_feature_returns_missing_neighbors(self):
        self.frame["temperature"] = np.nan
        targets = pd.DataFrame({"time": [0, 60], "nodeid": [1, 2]})
        result = nearest_observed(self.frame, "temperature", targets, 20)
        self.assertTrue(all(np.isnan(values).all() for values in result))

    def test_unknown_sensor_or_time_is_rejected(self):
        for time, sensor in [(30, 99), (90, 2)]:
            with self.subTest(time=time, sensor=sensor):
                with self.assertRaisesRegex(ValueError, "Unknown target"):
                    nearest_observed(self.frame, "temperature",
                                     pd.DataFrame({"time": [time], "nodeid": [sensor]}), 20)


class AnomalyTests(unittest.TestCase):
    def test_saved_experiment_f1(self):
        directory = ROOT / "anomaly_detection/results/injection"
        detections = load_detections(directory / "injected_detections.csv")
        with contextlib.redirect_stdout(io.StringIO()):
            metrics = evaluate_labels(detections, directory / "labels.csv").set_index("detector")
        self.assertEqual(metrics.loc["combined", "detected"], 31)
        self.assertEqual(metrics.loc["combined", "control_alerts"], 2)
        self.assertAlmostEqual(metrics.loc["combined", "f1_pct"], 200 * 31 / (165 + 33))

    def test_zero_detections_has_zero_f1(self):
        directory = ROOT / "anomaly_detection/results/injection"
        detections = load_detections(directory / "injected_detections.csv")
        detections[["rule_flag", "residual_flag", "is_anomaly"]] = False
        with contextlib.redirect_stdout(io.StringIO()):
            metrics = evaluate_labels(detections, directory / "labels.csv")
        self.assertTrue(metrics.f1_pct.eq(0).all())

    def test_f1_without_controls_is_not_reported(self):
        directory = ROOT / "anomaly_detection/results/injection"
        detections = load_detections(directory / "injected_detections.csv")
        labels = pd.read_csv(directory / "labels.csv")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "labels.csv"
            labels[labels.is_injected].to_csv(path, index=False)
            with contextlib.redirect_stdout(io.StringIO()):
                metrics = evaluate_labels(detections, path)
        self.assertTrue(metrics.f1_pct.isna().all())

    def test_targets_are_hidden_without_mutating_shared_context(self):
        class InspectInputs(torch.nn.Module):
            def forward(self, inputs):
                windows = inputs.reshape(-1, 40, 5, 8)
                for row, step, feature in [(0, 0, 0), (1, 39, 3)]:
                    assert windows[row, step, 0, feature] == 0
                    assert windows[row, step, 0, feature + 4] == 0
                    assert windows[row, step, 1, feature] == 1
                return torch.zeros((len(inputs), 40, 4))

        values = np.ones((40, 5, 4), dtype=np.float32)
        available = np.ones_like(values, dtype=bool)
        targets = pd.DataFrame({"time": [0, 1170], "nodeid": [0, 0],
                                "feature": ["temperature", "voltage"]})
        predicted = predict_targets(
            InspectInputs(), values, available, np.arange(40) * 30, np.arange(5),
            np.tile(np.arange(5), (5, 1)), np.zeros(4), np.ones(4), targets,
        )
        np.testing.assert_array_equal(predicted, [0, 0])
        self.assertTrue((values == 1).all())
        self.assertTrue(available.all())


if __name__ == "__main__":
    unittest.main()
