"""Leakage-safe continuous health labels derived from RUL."""

from __future__ import annotations

import numpy as np


def rul_to_health(rul: np.ndarray, reference_max_rul: float) -> np.ndarray:
    """Map RUL to [0, 1], using a training-only reference maximum."""
    values = np.asarray(rul, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("rul must be a finite 1-D array.")
    if not np.isfinite(reference_max_rul) or reference_max_rul <= 0:
        raise ValueError("reference_max_rul must be positive and finite.")
    return np.clip(values / reference_max_rul, 0.0, 1.0)


def health_state(health: np.ndarray) -> np.ndarray:
    """Create interpretable states without replacing continuous health."""
    values = np.asarray(health, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("health must be a finite 1-D array.")
    return np.select(
        [values >= 0.75, values >= 0.50, values >= 0.25],
        ["Healthy", "Degrading", "Poor"],
        default="Critical",
    )
