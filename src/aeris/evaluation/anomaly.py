"""Leakage-safe statistical anomaly scoring and persistence helpers."""

from __future__ import annotations

import numpy as np


def fit_robust_baseline(X_train: np.ndarray) -> dict[str, np.ndarray]:
    """Fit per-sensor robust location and scale from normal training windows."""
    values = np.asarray(X_train, dtype=float)
    flattened = values.reshape(-1, values.shape[-1])
    median = np.median(flattened, axis=0)
    mad = np.median(np.abs(flattened - median), axis=0)
    scale = np.maximum(1.4826 * mad, np.std(flattened, axis=0) * 0.1)
    return {"median": median, "scale": np.maximum(scale, 1e-6)}


def robust_sensor_scores(X: np.ndarray, baseline: dict[str, np.ndarray]) -> np.ndarray:
    """Return maximum robust z-score per sensor for each temporal window."""
    values = np.asarray(X, dtype=float)
    z = np.abs((values - baseline["median"]) / baseline["scale"])
    return np.max(z, axis=1)


def calibrate_threshold(scores: np.ndarray, percentile: float = 95.0) -> float:
    """Calibrate a global score threshold on normal calibration windows."""
    return float(np.percentile(np.asarray(scores, dtype=float), percentile))


def persistence_flags(point_flags: np.ndarray, consecutive: int) -> np.ndarray:
    """Require a fixed number of consecutive anomalous points."""
    flags = np.asarray(point_flags, dtype=bool)
    if consecutive < 1:
        raise ValueError("consecutive must be positive.")
    result = np.zeros_like(flags)
    run = 0
    for index, value in enumerate(flags):
        run = run + 1 if value else 0
        if run >= consecutive:
            result[index] = True
    return result
