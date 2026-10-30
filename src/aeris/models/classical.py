"""Classical regression models for flattened temporal RUL windows."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


def flatten_temporal_windows(X: np.ndarray) -> np.ndarray:
    """Return a copy of ``(samples, timesteps, features)`` as 2-D features."""
    values = np.asarray(X)
    if values.ndim != 3:
        raise ValueError(
            f"Expected a 3-D temporal array, received shape {values.shape}."
        )
    if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
        raise ValueError("Temporal features must contain only finite numeric values.")
    return values.reshape(values.shape[0], -1).copy()


class MeanRULRegressor:
    """A scikit-learn-compatible constant predictor using the training mean."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MeanRULRegressor":
        targets = np.asarray(y, dtype=float)
        if targets.ndim != 1 or targets.size == 0:
            raise ValueError("Training targets must be a non-empty 1-D array.")
        if not np.isfinite(targets).all():
            raise ValueError("Training targets must contain only finite values.")
        if len(X) != len(targets):
            raise ValueError("Features and targets must contain the same number of rows.")
        self.mean_ = float(np.mean(targets))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "mean_"):
            raise RuntimeError("MeanRULRegressor must be fitted before prediction.")
        return np.full(len(X), self.mean_, dtype=float)


def build_classical_models(random_state: int = 42) -> dict[str, Any]:
    """Build the required baselines and optional XGBoost model."""
    models: dict[str, Any] = {
        "mean_rul": MeanRULRegressor(),
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=100,
            random_state=random_state,
            n_jobs=-1,
        ),
    }
    try:
        from xgboost import XGBRegressor
    except ImportError:
        return models

    models["xgboost"] = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=random_state,
        n_jobs=-1,
    )
    return models
