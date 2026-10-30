"""Adapter for the existing leakage-safe C-MAPSS prepared arrays."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_prepared_cmapss(path: str | Path) -> dict[str, np.ndarray]:
    """Load the existing prepared arrays without refitting or transforming them."""
    directory = Path(path)
    result = {name: np.load(directory / f"{name}.npy") for name in ("X_train", "y_train", "X_validation", "y_validation")}
    if result["X_train"].ndim != 3 or result["X_validation"].ndim != 3:
        raise ValueError("Prepared C-MAPSS features must be 3-D.")
    return result
