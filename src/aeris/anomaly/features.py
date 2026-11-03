"""Leakage-safe temporal anomaly features."""

from __future__ import annotations

import numpy as np


def validate_windows(windows: np.ndarray) -> np.ndarray:
    """Validate and return finite temporal windows."""
    values = np.asarray(windows, dtype=float)
    if values.ndim != 3 or not np.isfinite(values).all():
        raise ValueError("windows must be a finite 3-D array.")
    return values


def fit_robust_baseline(windows: np.ndarray) -> dict[str, np.ndarray]:
    """Fit per-sensor median/MAD statistics on training windows only."""
    values = validate_windows(windows).reshape(-1, windows.shape[-1])
    median = np.median(values, axis=0)
    mad = np.median(np.abs(values - median), axis=0)
    scale = np.maximum(1.4826 * mad, np.std(values, axis=0) * 0.1)
    return {"median": median, "scale": np.maximum(scale, 1e-6)}


def sensor_anomaly_scores(
    windows: np.ndarray,
    baseline: dict[str, np.ndarray],
) -> np.ndarray:
    """Return one robust deviation score per sample and sensor."""
    values = validate_windows(windows)
    median = np.asarray(baseline["median"], dtype=float)
    scale = np.asarray(baseline["scale"], dtype=float)
    if median.shape != (values.shape[-1],) or scale.shape != median.shape:
        raise ValueError("baseline must contain one median and scale per sensor.")
    return np.max(np.abs((values - median) / np.maximum(scale, 1e-6)), axis=1)


def system_anomaly_scores(sensor_scores: np.ndarray, top_k: int = 3) -> np.ndarray:
    """Aggregate sensor evidence while requiring multivariate temporal evidence."""
    scores = np.asarray(sensor_scores, dtype=float)
    if scores.ndim != 2 or not np.isfinite(scores).all():
        raise ValueError("sensor_scores must be a finite 2-D array.")
    if not 1 <= top_k <= scores.shape[1]:
        raise ValueError("top_k must be between one and the sensor count.")
    top = np.sort(scores, axis=1)[:, -top_k:]
    return np.mean(top, axis=1)


def top_sensor_indices(sensor_scores: np.ndarray, top_k: int = 3) -> np.ndarray:
    """Return descending sensor indices contributing to each anomaly score."""
    scores = np.asarray(sensor_scores, dtype=float)
    if scores.ndim != 2 or not 1 <= top_k <= scores.shape[1]:
        raise ValueError("sensor_scores must be 2-D and top_k must be valid.")
    return np.argsort(scores, axis=1)[:, ::-1][:, :top_k]
