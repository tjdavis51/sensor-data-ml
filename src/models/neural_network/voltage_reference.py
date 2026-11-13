"""Same-sensor voltage interpolation restricted to each existing masked window."""

import numpy as np

from src.models.neural_network.data import WINDOW_LENGTH


def fit_fallbacks(training_values):
    voltage = training_values[:, :, 3]
    fallback = np.nanmedian(voltage, axis=0)
    global_median = np.nanmedian(voltage)
    if not np.isfinite(global_median):
        raise ValueError("No training voltage is available for fallback fitting")
    return np.where(np.isfinite(fallback), fallback, global_median).astype(np.float32)


def interpolate_voltage(values, available, positions, fallback):
    """Return standardized reference values and the number of visible readings."""
    steps = np.arange(values.shape[1])[None, :]
    before = np.where(available & (steps < positions[:, None]), steps, -1).max(axis=1)
    after = np.where(available & (steps > positions[:, None]), steps, values.shape[1]).min(axis=1)
    rows = np.arange(len(values))
    left = values[rows, np.maximum(before, 0)]
    right = values[rows, np.minimum(after, values.shape[1] - 1)]
    has_before = before >= 0
    has_after = after < values.shape[1]
    result = np.array(fallback, dtype=np.float32, copy=True)
    result[has_before] = left[has_before]
    result[has_after] = right[has_after]
    both = has_before & has_after
    weight = (positions[both] - before[both]) / (after[both] - before[both])
    result[both] = left[both] + weight * (right[both] - left[both])
    return result, available.sum(axis=1)


def build_reference(windows, fallbacks, means, stds):
    """Zero offsets for other outputs, voltage offsets in standardized units."""
    reference = np.zeros(len(windows), dtype=np.float32)
    visible_count = np.zeros(len(windows), dtype=np.int32)
    targets = np.flatnonzero(windows.features == 3)
    for start in range(0, len(targets), 512):
        indices = targets[start:start + 512]
        steps = windows.starts[indices, None] + np.arange(WINDOW_LENGTH)
        sensors = windows.sensors[indices]
        values = windows.values[steps, sensors[:, None], 3]
        available = windows.available[steps, sensors[:, None], 3]
        assert not available[np.arange(len(indices)), windows.positions[indices]].any()
        fallback = (fallbacks[sensors] - means[3]) / stds[3]
        reference[indices], visible_count[indices] = interpolate_voltage(
            values, available, windows.positions[indices], fallback
        )
    if not np.isfinite(reference).all():
        raise ValueError("Nonfinite voltage reference")
    return reference, visible_count
