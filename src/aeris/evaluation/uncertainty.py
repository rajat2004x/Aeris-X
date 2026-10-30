"""Reusable Monte Carlo Dropout uncertainty calculations."""

from __future__ import annotations

import numpy as np


def summarize_mc_predictions(mc_predictions: np.ndarray, interval_z: float = 1.96) -> dict[str, np.ndarray]:
    """Summarize an ``(samples, mc_samples)`` prediction matrix."""
    values = np.asarray(mc_predictions, dtype=float)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("MC predictions must be a 2-D array with at least two passes.")
    if not np.isfinite(values).all():
        raise ValueError("MC predictions must contain only finite values.")
    mean = values.mean(axis=1)
    std = values.std(axis=1, ddof=1)
    variance = std**2
    return {
        "mean_prediction": mean,
        "predictive_std": std,
        "predictive_variance": variance,
        "lower_interval": mean - interval_z * std,
        "upper_interval": mean + interval_z * std,
        "interval_width": 2 * interval_z * std,
    }


def relative_uncertainty(mean_prediction: np.ndarray, predictive_std: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """Return predictive standard deviation relative to prediction magnitude."""
    mean = np.asarray(mean_prediction, dtype=float)
    std = np.asarray(predictive_std, dtype=float)
    if mean.shape != std.shape:
        raise ValueError("Mean prediction and predictive standard deviation must have matching shapes.")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")
    return std / np.maximum(np.abs(mean), epsilon)


def interval_coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Return the fraction of targets contained by their prediction intervals."""
    actual, low, high = (np.asarray(value, dtype=float) for value in (y_true, lower, upper))
    if actual.ndim != 1 or actual.shape != low.shape or actual.shape != high.shape:
        raise ValueError("Coverage inputs must be matching 1-D arrays.")
    if not np.isfinite(actual).all() or not np.isfinite(low).all() or not np.isfinite(high).all():
        raise ValueError("Coverage inputs must contain only finite values.")
    return float(np.mean((actual >= low) & (actual <= high)))

