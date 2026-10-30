"""Reusable deterministic training loop for temporal RUL models."""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


def seed_everything(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch where supported."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device() -> torch.device:
    """Use CUDA when available and otherwise fall back to CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer:
    """Train a regression model and retain the best validation checkpoint."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        device: torch.device,
        checkpoint_path: str | Path,
        max_norm: float = 1.0,
        patience: int = 5,
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.checkpoint_path = Path(checkpoint_path)
        self.max_norm = max_norm
        self.patience = patience

    def _epoch(self, loader: DataLoader[Any], training: bool) -> float:
        self.model.train(training)
        total = 0.0
        count = 0
        for features, targets in loader:
            features = features.to(self.device)
            targets = targets.to(self.device)
            if training:
                self.optimizer.zero_grad(set_to_none=True)
            predictions = self.model(features)
            loss = self.loss_fn(predictions, targets)
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
                self.optimizer.step()
            total += loss.item() * len(targets)
            count += len(targets)
        return total / count

    def fit(self, train_loader: DataLoader[Any], validation_loader: DataLoader[Any], epochs: int) -> dict[str, list[float] | float]:
        if epochs < 1:
            raise ValueError("epochs must be positive.")
        history: dict[str, list[float] | float] = {"train_loss": [], "validation_loss": []}
        best_loss = float("inf")
        stale_epochs = 0
        started = time.perf_counter()
        for _ in range(epochs):
            train_loss = self._epoch(train_loader, True)
            with torch.no_grad():
                validation_loss = self._epoch(validation_loader, False)
            history["train_loss"].append(train_loss)
            history["validation_loss"].append(validation_loss)
            if validation_loss < best_loss:
                best_loss = validation_loss
                stale_epochs = 0
                self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(self.model.state_dict(), self.checkpoint_path)
            else:
                stale_epochs += 1
                if stale_epochs >= self.patience:
                    break
        self.model.load_state_dict(torch.load(self.checkpoint_path, map_location=self.device, weights_only=True))
        history["training_time_seconds"] = time.perf_counter() - started
        history["best_validation_loss"] = best_loss
        return history

    def predict(self, loader: DataLoader[Any]) -> tuple[np.ndarray, float]:
        self.model.eval()
        predictions: list[np.ndarray] = []
        started = time.perf_counter()
        with torch.no_grad():
            for features, _ in loader:
                predictions.append(self.model(features.to(self.device)).cpu().numpy())
        return np.concatenate(predictions), time.perf_counter() - started
