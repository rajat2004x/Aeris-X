"""Health trajectory utilities."""

from __future__ import annotations

import numpy as np


def split_by_engine_order(owners: np.ndarray, development_fraction: float = 0.6) -> tuple[np.ndarray, np.ndarray, list[int], list[int]]:
    """Split validation windows by deterministic engine ownership."""
    values = np.asarray(owners)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("owners must be a non-empty 1-D array.")
    unique = sorted(int(value) for value in np.unique(values))
    if len(unique) < 2 or not 0 < development_fraction < 1:
        raise ValueError("at least two engines and a valid fraction are required.")
    cut = min(max(1, int(len(unique) * development_fraction)), len(unique) - 1)
    development = unique[:cut]
    final = unique[cut:]
    return (
        np.isin(values, development),
        np.isin(values, final),
        development,
        final,
    )


def early_late_errors(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Compare errors in high-health and low-health regimes."""
    y_true = np.asarray(actual, dtype=float)
    y_pred = np.asarray(predicted, dtype=float)
    early = y_true >= np.quantile(y_true, 0.75)
    late = y_true <= np.quantile(y_true, 0.25)
    return {
        "early_life_mae": float(np.mean(np.abs(y_pred[early] - y_true[early]))) if np.any(early) else 0.0,
        "late_life_mae": float(np.mean(np.abs(y_pred[late] - y_true[late]))) if np.any(late) else 0.0,
    }
