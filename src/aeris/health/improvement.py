"""Controlled Step 12 health-estimation improvement experiments."""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from aeris.faults.injection import inject_fault
from aeris.health.evaluation import regression_metrics, trajectory_metrics
from aeris.health.labels import rul_to_health
from aeris.health.trajectory import early_late_errors
from aeris.models.temporal import TransformerRegressor


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def lifecycle_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Report errors in fixed health regimes, independent of model predictions."""
    y_true = np.asarray(actual, dtype=float)
    y_pred = np.asarray(predicted, dtype=float)
    if y_true.ndim != 1 or y_pred.shape != y_true.shape:
        raise ValueError("actual and predicted must be aligned 1-D arrays.")
    masks = {
        "early_life": y_true >= 0.75,
        "mid_life": (y_true >= 0.25) & (y_true < 0.75),
        "late_life": y_true < 0.25,
    }
    return {
        f"{name}_mae": float(np.mean(np.abs(y_pred[mask] - y_true[mask]))) if np.any(mask) else 0.0
        for name, mask in masks.items()
    }


class LifecycleHuberLoss(nn.Module):
    """Huber loss with fixed, predeclared early/mid/late weights."""

    def __init__(self, delta: float = 0.08) -> None:
        super().__init__()
        self.delta = delta

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        error = torch.nn.functional.huber_loss(prediction, target, delta=self.delta, reduction="none")
        weights = torch.where(target >= 0.75, 2.0, torch.where(target >= 0.25, 1.25, 1.0))
        return torch.mean(error * weights)


def temporal_consistency_penalty(
    prediction: torch.Tensor,
    target: torch.Tensor,
    owners: torch.Tensor,
) -> torch.Tensor:
    """Penalize predicted health increases only on true degrading adjacent windows."""
    if len(prediction) < 2:
        return prediction.new_zeros(())
    same_engine = owners[1:] == owners[:-1]
    degrading = target[1:] <= target[:-1]
    valid = same_engine & degrading
    if not torch.any(valid):
        return prediction.new_zeros(())
    return torch.relu(prediction[1:] - prediction[:-1])[valid].mean()


def _fit_candidate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    owners: np.ndarray,
    candidate: str,
    checkpoint: Path,
    seed: int,
    epochs: int = 8,
) -> dict[str, Any]:
    seed_everything(seed)
    model = TransformerRegressor(input_size=X_train.shape[2], max_length=X_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = DataLoader(
        TensorDataset(
            torch.as_tensor(X_train, dtype=torch.float32),
            torch.as_tensor(y_train, dtype=torch.float32),
            torch.as_tensor(owners, dtype=torch.long),
        ),
        batch_size=256,
        shuffle=False,
    )
    huber = LifecycleHuberLoss()
    started = time.perf_counter()
    model.train()
    for _ in range(epochs):
        for features, targets, batch_owners in loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(features)
            if candidate == "lifecycle_huber":
                loss = huber(prediction, targets)
            elif candidate == "temporal_consistency":
                loss = nn.functional.mse_loss(prediction, targets)
                loss = loss + 0.10 * temporal_consistency_penalty(prediction, targets, batch_owners)
            else:
                raise ValueError(f"Unknown candidate: {candidate}")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint)
    return {"training_time_seconds": time.perf_counter() - started, "epochs": epochs}


def predict_transformer(X: np.ndarray, checkpoint: Path) -> np.ndarray:
    """Load a frozen Transformer checkpoint and predict continuous health."""
    values = np.asarray(X)
    if values.ndim != 3 or not np.isfinite(values).all():
        raise ValueError("X must be a finite 3-D temporal array.")
    model = TransformerRegressor(input_size=values.shape[2], max_length=values.shape[1])
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    with torch.no_grad():
        return model(torch.as_tensor(values, dtype=torch.float32)).numpy()


def evaluate_predictions(
    actual: np.ndarray,
    predicted: np.ndarray,
    owners: np.ndarray,
    split: str,
    model: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {"model": model, "split": split}
    result.update(regression_metrics(actual, predicted))
    result.update(trajectory_metrics(actual, predicted))
    result.update(lifecycle_metrics(actual, predicted))
    result.update(early_late_errors(actual, predicted))
    result["engines"] = int(len(np.unique(owners)))
    return result


def robustness_evaluation(
    X: np.ndarray,
    actual: np.ndarray,
    owners: np.ndarray,
    baseline_prediction: np.ndarray,
    candidate_prediction: np.ndarray,
    baseline_checkpoint: Path,
    candidate_checkpoint: Path,
    training_std: np.ndarray,
    reference_max_rul: float,
) -> list[dict[str, Any]]:
    """Evaluate clean-vs-corrupted health stability without retraining."""
    corrupted = np.asarray(X, dtype=np.float32).copy()
    for index in range(0, len(corrupted), 4):
        corrupted[index], _ = inject_fault(
            corrupted[index], 0, "BIAS", 0.5, 1.5, float(training_std[0]), seed=42 + index
        )
    baseline_corrupted = rul_to_health(
        predict_transformer(corrupted, baseline_checkpoint), reference_max_rul
    )
    candidate_corrupted = predict_transformer(corrupted, candidate_checkpoint)
    rows = []
    for name, clean, changed in (
        ("baseline_transformer", baseline_prediction, baseline_corrupted),
        ("selected_candidate", candidate_prediction, candidate_corrupted),
    ):
        healthy = actual >= 0.75
        rows.append({
            "model": name,
            "corruption": "sensor_0_bias_severity_1.5",
            "clean_mae": float(np.mean(np.abs(clean - actual))),
            "corrupted_mae": float(np.mean(np.abs(changed - actual))),
            "mae_degradation": float(np.mean(np.abs(changed - actual)) - np.mean(np.abs(clean - actual))),
            "prediction_stability": float(np.mean(np.abs(changed - clean))),
            "healthy_as_degraded_rate": float(np.mean(changed[healthy] < 0.5)) if np.any(healthy) else None,
            "healthy_windows": int(np.sum(healthy)),
            "affected_windows": int(len(range(0, len(corrupted), 4))),
        })
    return rows
