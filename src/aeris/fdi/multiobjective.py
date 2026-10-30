"""Small, deterministic utilities for multi-objective FDI selection."""

from __future__ import annotations

import numpy as np


def pareto_frontier(rows: list[dict[str, float]], maximize: set[str], minimize: set[str]) -> list[int]:
    """Return indices of non-dominated rows."""
    result = []
    for i, candidate in enumerate(rows):
        dominated = False
        for j, other in enumerate(rows):
            if i == j:
                continue
            at_least = False
            no_worse = True
            for key in maximize:
                if other[key] < candidate[key]:
                    no_worse = False
                if other[key] > candidate[key]:
                    at_least = True
            for key in minimize:
                if other[key] > candidate[key]:
                    no_worse = False
                if other[key] < candidate[key]:
                    at_least = True
            if no_worse and at_least:
                dominated = True
                break
        if not dominated:
            result.append(i)
    return result


def threshold_summary(labels: np.ndarray, scores: np.ndarray, thresholds: np.ndarray) -> list[dict[str, float]]:
    from sklearn.metrics import f1_score, precision_score, recall_score

    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    rows = []
    for threshold in thresholds:
        flags = scores >= threshold
        normal = labels == 0
        faults = labels == 1
        rows.append({
            "threshold": float(threshold),
            "precision": float(precision_score(labels, flags, zero_division=0)),
            "recall": float(recall_score(labels, flags, zero_division=0)),
            "f1": float(f1_score(labels, flags, zero_division=0)),
            "fpr": float(np.mean(flags[normal])) if np.any(normal) else 0.0,
            "fnr": float(np.mean(~flags[faults])) if np.any(faults) else 0.0,
        })
    return rows


def route_domain(distance: float, known_limit: float = 0.05, shifted_limit: float = 0.15) -> str:
    if distance <= known_limit:
        return "KNOWN_OPERATING_ENVELOPE"
    if distance <= shifted_limit:
        return "SHIFTED_OPERATING_ENVELOPE"
    return "UNKNOWN_AMBIGUOUS_ENVELOPE"


def fuse_scores(*scores: np.ndarray, weights: tuple[float, ...] | None = None) -> np.ndarray:
    if not scores:
        raise ValueError("at least one score is required")
    arrays = [np.asarray(value, dtype=float) for value in scores]
    if any(value.shape != arrays[0].shape for value in arrays):
        raise ValueError("score arrays must have identical shapes")
    if weights is None:
        weights = tuple(1.0 / len(arrays) for _ in arrays)
    if len(weights) != len(arrays):
        raise ValueError("weights must match score count")
    return np.sum(np.stack(arrays) * np.asarray(weights)[:, None], axis=0)


def reject_unknown(confidence: np.ndarray, threshold: float) -> np.ndarray:
    return np.asarray(confidence, dtype=float) < float(threshold)
