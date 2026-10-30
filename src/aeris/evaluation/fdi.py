"""Metrics and sensor isolation helpers for Step 10."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score


def reconstruction_scores(original: np.ndarray, reconstructed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    errors = (np.asarray(original) - np.asarray(reconstructed)) ** 2
    return errors.mean(axis=(1, 2)), errors.max(axis=1)


def threshold_from_normal(scores: np.ndarray, percentile: float = 95.0) -> float:
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or not 0 < percentile < 100:
        raise ValueError("scores must be 1-D and percentile must be in (0, 100).")
    return float(np.percentile(values, percentile))


def detection_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    actual = np.asarray(labels, dtype=int)
    predicted = np.asarray(scores) >= threshold
    result = {
        "precision": float(precision_score(actual, predicted, zero_division=0)),
        "recall": float(recall_score(actual, predicted, zero_division=0)),
        "f1": float(f1_score(actual, predicted, zero_division=0)),
        "accuracy": float(accuracy_score(actual, predicted)),
        "false_positive_rate": float(np.sum((predicted == 1) & (actual == 0)) / max(1, np.sum(actual == 0))),
        "false_negative_rate": float(np.sum((predicted == 0) & (actual == 1)) / max(1, np.sum(actual == 1))),
    }
    if len(np.unique(actual)) > 1:
        result["roc_auc"] = float(roc_auc_score(actual, scores))
        result["pr_auc"] = float(average_precision_score(actual, scores))
    else:
        result["roc_auc"] = float("nan")
        result["pr_auc"] = float("nan")
    return result


def isolation_rank(sensor_scores: np.ndarray, target_sensor: int) -> tuple[int, list[int]]:
    ranking = np.argsort(np.asarray(sensor_scores))[::-1].tolist()
    return ranking.index(target_sensor) + 1, ranking
