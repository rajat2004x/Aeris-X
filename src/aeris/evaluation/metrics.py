"""Regression metrics used by the RUL benchmark."""

from __future__ import annotations

import numpy as np


def _validate_targets(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    if actual.ndim != 1 or predicted.ndim != 1:
        raise ValueError("RUL targets and predictions must be 1-D arrays.")
    if actual.shape != predicted.shape or actual.size == 0:
        raise ValueError("RUL targets and predictions must have the same non-empty shape.")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("RUL targets and predictions must contain only finite values.")
    return actual, predicted


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    actual, predicted = _validate_targets(y_true, y_pred)
    return float(np.mean(np.abs(predicted - actual)))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    actual, predicted = _validate_targets(y_true, y_pred)
    return float(np.sqrt(np.mean((predicted - actual) ** 2)))


def phm_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate the NASA/PHM asymmetric RUL score.

    Under-prediction (negative ``prediction - actual``) uses a 13-cycle
    scale and over-prediction uses a 10-cycle scale.
    """
    actual, predicted = _validate_targets(y_true, y_pred)
    error = predicted - actual
    penalties = np.where(error < 0, np.exp(-error / 13.0) - 1.0, np.exp(error / 10.0) - 1.0)
    return float(np.sum(penalties))
