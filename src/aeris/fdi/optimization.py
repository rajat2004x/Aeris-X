"""Sensor-level representations and alarm policies for Step 10.2.1."""

from __future__ import annotations

import numpy as np


def sensor_feature_tensor(
    windows: np.ndarray,
    baseline_median: np.ndarray,
    baseline_scale: np.ndarray,
) -> np.ndarray:
    """Return one feature vector per sample and sensor."""
    values = np.nan_to_num(np.asarray(windows, dtype=float))
    if values.ndim != 3:
        raise ValueError("windows must have shape (samples, time, sensors)")
    median = np.asarray(baseline_median, dtype=float)
    scale = np.maximum(np.asarray(baseline_scale, dtype=float), 1e-6)
    if median.shape != (values.shape[2],) or scale.shape != (values.shape[2],):
        raise ValueError("baseline arrays must contain one value per sensor")
    residual = values - median[None, None, :]
    robust = residual / scale[None, None, :]
    derivative = np.diff(values, axis=1, prepend=values[:, :1, :])
    second = np.diff(derivative, axis=1, prepend=derivative[:, :1, :])
    mean = np.mean(values, axis=1)
    spread = np.std(values, axis=1)
    slope = values[:, -1, :] - values[:, 0, :]
    sharpness = np.max(np.abs(derivative), axis=1)
    persistence = np.mean(np.abs(robust) >= 3.0, axis=1)
    correlation = np.abs(residual - np.mean(residual, axis=2, keepdims=True))
    return np.stack(
        [
            np.mean(np.abs(residual), axis=1),
            np.max(np.abs(residual), axis=1),
            np.mean(np.abs(robust), axis=1),
            np.max(np.abs(robust), axis=1),
            np.mean(np.abs(derivative), axis=1),
            np.max(np.abs(second), axis=1),
            np.abs(mean),
            spread,
            slope,
            sharpness,
            persistence,
            np.max(correlation, axis=1),
        ],
        axis=2,
    )


def persistence_flags(scores: np.ndarray, threshold: float, window: int = 3, required: int = 2) -> np.ndarray:
    """Apply an N-of-M alarm rule to sample-ordered anomaly scores."""
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or window < 1 or not 1 <= required <= window:
        raise ValueError("scores must be 1-D and required must be within window")
    flags = values >= threshold
    result = np.zeros(len(values), dtype=bool)
    for index in range(len(values)):
        start = max(0, index - window + 1)
        result[index] = int(np.sum(flags[start : index + 1])) >= required
    return result


def sensor_set_metrics(predicted: list[set[int]], actual: list[set[int]]) -> dict[str, float]:
    """Calculate set precision, recall, and Jaccard for multi-sensor faults."""
    if len(predicted) != len(actual):
        raise ValueError("predicted and actual collections must have equal length")
    precisions = []
    recalls = []
    jaccards = []
    for guess, target in zip(predicted, actual):
        precisions.append(len(guess & target) / max(1, len(guess)))
        recalls.append(len(guess & target) / max(1, len(target)))
        jaccards.append(len(guess & target) / max(1, len(guess | target)))
    return {
        "set_precision": float(np.mean(precisions)),
        "set_recall": float(np.mean(recalls)),
        "jaccard": float(np.mean(jaccards)),
    }
