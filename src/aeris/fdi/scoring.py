"""Leakage-safe, sensor- and time-aware FDI features."""

from __future__ import annotations

import numpy as np


def _validate_windows(X: np.ndarray) -> np.ndarray:
    values = np.asarray(X, dtype=float)
    if values.ndim != 3:
        raise ValueError("telemetry windows must have shape (samples, time, sensors)")
    return values


def derivative_sensor_scores(X: np.ndarray) -> np.ndarray:
    """Return the maximum absolute first difference per sensor and window."""
    values = _validate_windows(X)
    differences = np.diff(values, axis=1, prepend=values[:, :1, :])
    return np.max(np.abs(differences), axis=1)


def cross_sensor_scores(X: np.ndarray, baseline: np.ndarray | None = None) -> np.ndarray:
    """Score sensor disagreement from a training-derived correlation baseline."""
    values = _validate_windows(X)
    centered = values - np.mean(values, axis=2, keepdims=True)
    disagreement = np.abs(centered)
    if baseline is not None:
        expected = np.asarray(baseline, dtype=float)
        if expected.shape != (values.shape[2],):
            raise ValueError("baseline must contain one value per sensor")
        disagreement = disagreement / np.maximum(expected[None, None, :], 1e-6)
    return np.max(disagreement, axis=1)


def temporal_persistence(sensor_scores: np.ndarray, threshold: float, consecutive: int = 3) -> np.ndarray:
    """Return the longest above-threshold run per sensor and sample."""
    scores = np.asarray(sensor_scores, dtype=float)
    if scores.ndim != 3 or consecutive < 1:
        raise ValueError("sensor_scores must be 3-D and consecutive must be positive")
    flags = scores >= threshold
    result = np.zeros((scores.shape[0], scores.shape[2]), dtype=float)
    for sample in range(scores.shape[0]):
        run = np.zeros(scores.shape[2], dtype=int)
        for timestep in range(scores.shape[1]):
            run = np.where(flags[sample, timestep], run + 1, 0)
            result[sample] = np.maximum(result[sample], run)
    return result


def build_feature_matrix(
    X: np.ndarray,
    robust_sensor: np.ndarray,
    derivative_sensor: np.ndarray,
    cross_sensor: np.ndarray,
) -> np.ndarray:
    """Create finite per-window features for second-stage classification."""
    values = _validate_windows(X)
    if any(np.asarray(item).shape != (len(values), values.shape[2]) for item in (robust_sensor, derivative_sensor, cross_sensor)):
        raise ValueError("all sensor feature arrays must have shape (samples, sensors)")
    level = np.mean(values, axis=1)
    slope = values[:, -1, :] - values[:, 0, :]
    spread = np.std(values, axis=1)
    return np.concatenate(
        [robust_sensor, derivative_sensor, cross_sensor, level, slope, spread],
        axis=1,
    )
