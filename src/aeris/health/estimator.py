"""Reusable estimators for health-state regression."""

from __future__ import annotations

import numpy as np


def validate_regression_inputs(features: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(features, dtype=float)
    labels = np.asarray(targets, dtype=float)
    if values.ndim != 2 or labels.ndim != 1 or len(values) != len(labels):
        raise ValueError("features must be 2-D and targets must be aligned 1-D.")
    if not np.isfinite(values).all() or not np.isfinite(labels).all():
        raise ValueError("features and targets must be finite.")
    return values, labels


class MeanHealthEstimator:
    """Constant health baseline fitted on training health only."""

    def fit(self, features: np.ndarray, targets: np.ndarray) -> "MeanHealthEstimator":
        _, labels = validate_regression_inputs(features, targets)
        if not len(labels):
            raise ValueError("targets must not be empty.")
        self.value_ = float(np.mean(labels))
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if not hasattr(self, "value_"):
            raise RuntimeError("Estimator must be fitted before prediction.")
        return np.full(len(features), self.value_, dtype=float)
