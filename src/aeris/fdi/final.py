"""Lightweight leakage-safe components for the final Step 10.4 FDI run."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge


def temporal_features(windows: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(np.asarray(windows, dtype=float))
    first = np.diff(values, axis=1, prepend=values[:, :1])
    second = np.diff(first, axis=1, prepend=first[:, :1])
    return np.stack(
        [
            np.mean(np.abs(values), axis=1),
            np.std(values, axis=1),
            np.mean(np.abs(first), axis=1),
            np.max(np.abs(second), axis=1),
            values[:, -1] - values[:, 0],
        ],
        axis=2,
    )


def sensor_scores(windows: np.ndarray, medians: np.ndarray, scales: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(np.asarray(windows, dtype=float))
    z = np.abs((values - medians[None, None, :]) / np.maximum(scales[None, None, :], 1e-6))
    return np.max(z, axis=1)


def soft_scores(local_scores: np.ndarray) -> np.ndarray:
    point = np.clip(np.asarray(local_scores, dtype=float) / 12.0, 0.0, 1.0)
    return 0.65 * np.max(point, axis=1) + 0.35 * np.mean(point, axis=1)


def fit_peer_models(windows: np.ndarray) -> list[Ridge]:
    values = np.asarray(windows, dtype=float).reshape(-1, windows.shape[-1])
    models = []
    for target in range(values.shape[1]):
        peers = np.delete(values, target, axis=1)
        model = Ridge(alpha=1.0)
        model.fit(peers, values[:, target])
        models.append(model)
    return models


def peer_residual_scores(windows: np.ndarray, models: list[Ridge]) -> np.ndarray:
    values = np.asarray(windows, dtype=float)
    flat = values.reshape(-1, values.shape[-1])
    output = np.zeros_like(flat)
    for target, model in enumerate(models):
        prediction = model.predict(np.delete(flat, target, axis=1))
        output[:, target] = np.abs(flat[:, target] - prediction)
    return np.max(output.reshape(values.shape), axis=1)


def expected_calibration_error(confidence: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    return float(
        sum(
            np.mean(mask) * abs(np.mean(confidence[mask]) - np.mean(correct[mask]))
            for lower, upper in zip(edges[:-1], edges[1:])
            if np.any(mask := ((confidence >= lower) & (confidence <= upper if upper == 1 else confidence < upper)))
        )
    )
