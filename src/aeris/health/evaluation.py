"""Health estimation metrics and trajectory diagnostics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import r2_score
from scipy.stats import spearmanr


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(actual, dtype=float)
    y_pred = np.asarray(predicted, dtype=float)
    if y_true.ndim != 1 or y_pred.shape != y_true.shape or not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError("actual and predicted must be aligned finite 1-D arrays.")
    errors = y_pred - y_true
    correlation = float(np.corrcoef(y_true, y_pred)[0, 1]) if np.std(y_true) and np.std(y_pred) else 0.0
    rho = (
        spearmanr(y_true, y_pred).statistic
        if np.unique(y_true).size > 1 and np.unique(y_pred).size > 1
        else 0.0
    )
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "r2": float(r2_score(y_true, y_pred)),
        "correlation": correlation,
        "spearman_correlation": float(rho) if np.isfinite(rho) else 0.0,
    }


def trajectory_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    """Measure smoothness and non-increasing health consistency."""
    metrics = regression_metrics(actual, predicted)
    y_true = np.asarray(actual, dtype=float)
    y_pred = np.asarray(predicted, dtype=float)
    true_delta = np.diff(y_true)
    pred_delta = np.diff(y_pred)
    violations = int(np.sum(pred_delta > 0))
    return {
        **metrics,
        "smoothness_mean_abs_delta": float(np.mean(np.abs(pred_delta))) if len(pred_delta) else 0.0,
        "monotonicity_violations": violations,
        "monotonicity_violation_rate": violations / max(1, len(pred_delta)),
        "degradation_direction_consistency": float(np.mean(np.sign(true_delta) == np.sign(pred_delta))) if len(pred_delta) else 1.0,
    }
