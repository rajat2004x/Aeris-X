"""Sensor-localized features and open-set utilities for Step 10.2.3."""

from __future__ import annotations

import numpy as np


def sensor_feature_tensor(
    windows: np.ndarray,
    median: np.ndarray,
    scale: np.ndarray,
    peer_residual: np.ndarray | None = None,
) -> np.ndarray:
    """Return causal features with shape (samples, sensors, features)."""
    values = np.nan_to_num(np.asarray(windows, dtype=float))
    if values.ndim != 3:
        raise ValueError("windows must have shape (samples, time, sensors)")
    median = np.asarray(median, dtype=float)
    scale = np.maximum(np.asarray(scale, dtype=float), 1e-6)
    if median.shape != (values.shape[2],) or scale.shape != (values.shape[2],):
        raise ValueError("baseline arrays must contain one value per sensor")
    residual = values - median[None, None, :]
    robust = residual / scale[None, None, :]
    first = np.diff(values, axis=1, prepend=values[:, :1])
    second = np.diff(first, axis=1, prepend=first[:, :1])
    abnormal = np.abs(robust) >= 3
    ewma = np.cumsum(residual, axis=1) / np.arange(1, values.shape[1] + 1)[None, :, None]
    cumulative = np.cumsum(np.sign(robust) * np.maximum(np.abs(robust) - 1, 0), axis=1)
    disagreement = residual - np.mean(residual, axis=2, keepdims=True)
    features = [
        np.mean(residual, axis=1),
        np.mean(np.abs(residual), axis=1),
        np.max(np.abs(residual), axis=1),
        np.mean(np.abs(robust), axis=1),
        np.max(np.abs(robust), axis=1),
        np.mean(first, axis=1),
        np.max(np.abs(second), axis=1),
        values[:, -1] - values[:, 0],
        np.mean(np.abs(residual), axis=1),
        np.var(values, axis=1),
        np.max(np.abs(residual), axis=1),
        np.median(np.abs(residual), axis=1),
        np.mean(abnormal, axis=1),
        np.max(abnormal, axis=1),
        np.max(np.abs(first), axis=1),
        ewma[:, -1],
        cumulative[:, -1],
        np.max(np.abs(disagreement), axis=1),
    ]
    if peer_residual is not None:
        peer = np.asarray(peer_residual, dtype=float)
        if peer.shape != (len(values), values.shape[2]):
            raise ValueError("peer_residual must have shape (samples, sensors)")
        features.extend([peer, np.abs(peer)])
    return np.stack(features, axis=2)


def standardized_distance(features: np.ndarray, centroids: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Return nearest-class standardized distance and class index."""
    values = np.asarray(features, dtype=float)
    names = list(centroids)
    matrix = np.stack([np.linalg.norm(values - centroids[name], axis=1) for name in names], axis=1)
    nearest = np.argmin(matrix, axis=1)
    return matrix[np.arange(len(values)), nearest], nearest


def expected_calibration_error(confidence: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    """Calculate absolute reliability gap across confidence bins."""
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    error = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (confidence >= lower) & (confidence <= upper if upper == 1 else confidence < upper)
        if np.any(mask):
            error += np.mean(mask) * abs(float(np.mean(confidence[mask])) - float(np.mean(correct[mask])))
    return float(error)
