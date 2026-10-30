"""Condition-aware normalization and robust baseline utilities."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans


def fit_condition_model(
    condition_rows: np.ndarray, n_conditions: int = 6, seed: int = 42
) -> dict[str, object]:
    """Fit condition clusters using normal training telemetry only."""
    values = np.asarray(condition_rows, dtype=float)
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("condition_rows must be a two-dimensional array")
    mean = values.mean(axis=0)
    scale = np.maximum(values.std(axis=0), 1e-8)
    normalized = (values - mean) / scale
    count = min(max(1, n_conditions), len(values))
    model = KMeans(n_clusters=count, random_state=seed, n_init=10)
    model.fit(normalized)
    return {"mean": mean, "scale": scale, "model": model}


def assign_conditions(rows: np.ndarray, model: dict[str, object]) -> np.ndarray:
    values = np.asarray(rows, dtype=float)
    normalized = (values - model["mean"]) / model["scale"]
    return model["model"].predict(normalized)


def fit_condition_baseline(
    windows: np.ndarray,
    condition_ids: np.ndarray,
) -> dict[str, object]:
    """Fit global and per-condition robust sensor baselines on normal data."""
    values = np.asarray(windows, dtype=float)
    conditions = np.asarray(condition_ids, dtype=int)
    if values.ndim != 3 or len(values) != len(conditions):
        raise ValueError("windows and condition_ids must describe the same samples")
    flat = values.reshape(-1, values.shape[-1])
    global_median = np.median(flat, axis=0)
    global_mad = np.median(np.abs(flat - global_median), axis=0)
    global_scale = np.maximum(1.4826 * global_mad, np.std(flat, axis=0) * 0.1)
    medians: dict[int, np.ndarray] = {}
    scales: dict[int, np.ndarray] = {}
    for condition in np.unique(conditions):
        subset = values[conditions == condition].reshape(-1, values.shape[-1])
        median = np.median(subset, axis=0)
        mad = np.median(np.abs(subset - median), axis=0)
        medians[int(condition)] = median
        scales[int(condition)] = np.maximum(
            np.maximum(1.4826 * mad, np.std(subset, axis=0) * 0.1),
            1e-6,
        )
    return {
        "global_median": global_median,
        "global_scale": np.maximum(global_scale, 1e-6),
        "medians": medians,
        "scales": scales,
    }


def robust_window_scores(
    windows: np.ndarray,
    baseline: dict[str, object],
    condition_ids: np.ndarray | None = None,
    condition_aware: bool = False,
) -> np.ndarray:
    """Return maximum temporal robust-z score per window."""
    values = np.asarray(windows, dtype=float)
    if not condition_aware or condition_ids is None:
        z = np.abs((values - baseline["global_median"]) / baseline["global_scale"])
        return np.max(z, axis=(1, 2))
    conditions = np.asarray(condition_ids, dtype=int)
    scores = np.zeros(len(values), dtype=float)
    for condition in np.unique(conditions):
        mask = conditions == condition
        median = baseline["medians"].get(int(condition), baseline["global_median"])
        scale = baseline["scales"].get(int(condition), baseline["global_scale"])
        scores[mask] = np.max(np.abs((values[mask] - median) / scale), axis=(1, 2))
    return scores
