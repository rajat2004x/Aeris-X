"""Evidence feature and calibration helpers for Step 10.4.1."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import brier_score_loss


def evidence_features(
    local: np.ndarray,
    peer: np.ndarray,
    temporal: np.ndarray,
    condition: np.ndarray,
    domain: np.ndarray,
) -> np.ndarray:
    local = np.asarray(local, dtype=float)
    peer = np.asarray(peer, dtype=float)
    temporal = np.asarray(temporal, dtype=float)
    condition = np.asarray(condition, dtype=float).reshape(-1, 1)
    domain = np.asarray(domain, dtype=float).reshape(-1, 1)
    local_order = np.sort(local, axis=1)[:, ::-1]
    peer_order = np.sort(peer, axis=1)[:, ::-1]
    top = local_order[:, 0]
    return np.column_stack(
        [
            np.max(local, axis=1),
            np.mean(local, axis=1),
            np.std(local, axis=1),
            top,
            top / np.maximum(np.sum(local, axis=1), 1e-6),
            np.mean(local_order[:, :3], axis=1),
            np.max(peer, axis=1),
            np.mean(peer, axis=1),
            np.max(temporal, axis=1),
            np.mean(temporal, axis=1),
            condition,
            domain,
        ]
    )


def calibration_metrics(probability: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    probability = np.asarray(probability, dtype=float)
    labels = np.asarray(labels, dtype=int)
    edges = np.linspace(0, 1, 11)
    ece = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (probability >= low) & (
            probability <= high if high == 1 else probability < high
        )
        if np.any(mask):
            ece += np.mean(mask) * abs(np.mean(probability[mask]) - np.mean(labels[mask]))
    return {"brier": float(brier_score_loss(labels, probability)), "ece": float(ece)}
