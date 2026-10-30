"""Leakage-safe condition-residual features for Step 10.4.2."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge


def context_rows(windows: np.ndarray, settings: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(np.asarray(windows, dtype=float))
    context = np.asarray(settings, dtype=float)
    if values.ndim != 3 or context.shape[0] != values.shape[0]:
        raise ValueError("windows and settings have incompatible shapes")
    progress = np.linspace(0.0, 1.0, values.shape[1], dtype=float)
    progress = np.broadcast_to(progress[None, :, None], values.shape[:2] + (1,))
    settings_3d = np.broadcast_to(context[:, None, :], (len(values), values.shape[1], context.shape[1]))
    return np.concatenate((settings_3d, progress), axis=2)


def fit_residual_models(windows: np.ndarray, settings: np.ndarray) -> dict[str, object]:
    """Fit one context predictor per sensor using normal training windows only."""
    values = np.nan_to_num(np.asarray(windows, dtype=float))
    rows = context_rows(values, settings)
    flat_x = rows.reshape(-1, rows.shape[-1])
    models = []
    for sensor in range(values.shape[-1]):
        model = Ridge(alpha=1.0)
        model.fit(flat_x, values[:, :, sensor].reshape(-1))
        models.append(model)
    predicted = np.stack(
        [model.predict(flat_x).reshape(values.shape[:2]) for model in models],
        axis=2,
    )
    residual = values - predicted
    median = np.median(residual.reshape(-1, residual.shape[-1]), axis=0)
    mad = np.median(
        np.abs(residual.reshape(-1, residual.shape[-1]) - median), axis=0
    )
    scale = np.maximum(1.4826 * mad, np.std(residual, axis=(0, 1)) * 0.1)
    return {"models": models, "scale": np.maximum(scale, 1e-6)}


def residual_windows(
    windows: np.ndarray, settings: np.ndarray, model: dict[str, object]
) -> np.ndarray:
    values = np.nan_to_num(np.asarray(windows, dtype=float))
    rows = context_rows(values, settings)
    flat_x = rows.reshape(-1, rows.shape[-1])
    predicted = np.stack(
        [
            fitted.predict(flat_x).reshape(values.shape[:2])
            for fitted in model["models"]
        ],
        axis=2,
    )
    return values - predicted


def residual_features(residuals: np.ndarray, scale: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    residual = np.nan_to_num(np.asarray(residuals, dtype=float))
    robust = residual / np.maximum(np.asarray(scale)[None, None, :], 1e-6)
    first = np.diff(robust, axis=1, prepend=robust[:, :1])
    abnormal = np.abs(robust) >= 3.0
    local = np.stack(
        (
            np.max(np.abs(robust), axis=1),
            np.mean(np.abs(robust), axis=1),
            np.mean(np.abs(first), axis=1),
            np.max(np.abs(first), axis=1),
            np.mean(abnormal, axis=1),
            np.max(abnormal, axis=1),
            np.abs(robust[:, -1] - robust[:, 0]),
        ),
        axis=2,
    )
    peer = np.abs(robust - np.mean(robust, axis=2, keepdims=True))
    evidence = 0.45 * local[:, :, 0] / 12.0
    evidence += 0.20 * local[:, :, 2] / 6.0
    evidence += 0.20 * local[:, :, 4]
    evidence += 0.15 * np.max(peer, axis=1) / 12.0
    return np.clip(evidence, 0.0, 1.0), local
