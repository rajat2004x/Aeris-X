"""Point-level anomaly evaluation metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score


def detection_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    """Calculate point-level metrics at a frozen threshold."""
    actual = np.asarray(labels, dtype=int)
    values = np.asarray(scores, dtype=float)
    if actual.ndim != 1 or values.shape != actual.shape or not np.isfinite(values).all():
        raise ValueError("labels and scores must be aligned finite 1-D arrays.")
    predicted = values >= threshold
    negatives = max(1, int(np.sum(actual == 0)))
    positives = max(1, int(np.sum(actual == 1)))
    return {
        "precision": float(precision_score(actual, predicted, zero_division=0)),
        "recall": float(recall_score(actual, predicted, zero_division=0)),
        "f1": float(f1_score(actual, predicted, zero_division=0)),
        "fpr": float(np.sum(predicted & (actual == 0)) / negatives),
        "fnr": float(np.sum(~predicted & (actual == 1)) / positives),
        "roc_auc": float(roc_auc_score(actual, values)) if len(np.unique(actual)) > 1 else None,
        "pr_auc": float(average_precision_score(actual, values)) if len(np.unique(actual)) > 1 else None,
    }


def percentile_threshold(normal_scores: np.ndarray, percentile: float) -> float:
    """Select a threshold from normal development scores only."""
    values = np.asarray(normal_scores, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or not 0 < percentile < 100:
        raise ValueError("normal_scores must be finite 1-D and percentile must be in (0, 100).")
    return float(np.percentile(values, percentile))
