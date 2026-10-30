"""Temporal adaptive FDI helpers for the Step 10.2.2 experiment."""

from __future__ import annotations

import numpy as np


def temporal_sensor_features(
    windows: np.ndarray,
    median: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    """Build causal per-sensor features from each telemetry window."""
    values = np.nan_to_num(np.asarray(windows, dtype=float))
    if values.ndim != 3:
        raise ValueError("windows must be three-dimensional")
    residual = values - median[None, None, :]
    robust = residual / np.maximum(scale[None, None, :], 1e-6)
    first = np.diff(values, axis=1, prepend=values[:, :1])
    second = np.diff(first, axis=1, prepend=first[:, :1])
    abnormal = np.abs(robust) >= 3.0
    return np.stack(
        (
            np.mean(np.abs(residual), axis=1),
            np.max(np.abs(residual), axis=1),
            np.mean(np.abs(robust), axis=1),
            np.max(np.abs(robust), axis=1),
            np.mean(np.abs(first), axis=1),
            np.max(np.abs(second), axis=1),
            np.mean(values, axis=1),
            np.std(values, axis=1),
            values[:, -1] - values[:, 0],
            np.mean(abnormal, axis=1),
            np.max(abnormal, axis=1),
            np.max(np.abs(residual - np.mean(residual, axis=2, keepdims=True)), axis=1),
        ),
        axis=2,
    )


def soft_evidence(features: np.ndarray) -> np.ndarray:
    """Combine complementary sensor features without a hard persistence gate."""
    values = np.asarray(features, dtype=float)
    robust = np.clip(values[:, :, 3] / 12.0, 0.0, 1.0)
    derivative = np.clip(values[:, :, 4] / 6.0, 0.0, 1.0)
    persistence = np.clip(values[:, :, 9], 0.0, 1.0)
    disagreement = np.clip(values[:, :, 11] / 12.0, 0.0, 1.0)
    return 0.45 * robust + 0.2 * derivative + 0.15 * persistence + 0.2 * disagreement


def ewma(values: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    """Apply a causal EWMA to an ordered score series."""
    scores = np.asarray(values, dtype=float)
    if scores.ndim != 1 or not 0 < alpha <= 1:
        raise ValueError("values must be one-dimensional and alpha must be in (0, 1]")
    result = np.empty_like(scores)
    if len(scores) == 0:
        return result
    result[0] = scores[0]
    for index in range(1, len(scores)):
        result[index] = alpha * scores[index] + (1 - alpha) * result[index - 1]
    return result
