"""Dataset utilities for prepared temporal windows."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class TemporalDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Expose finite ``(samples, timesteps, features)`` arrays as tensors."""

    def __init__(self, features: np.ndarray, targets: np.ndarray) -> None:
        values = np.asarray(features)
        labels = np.asarray(targets)
        if values.ndim != 3:
            raise ValueError("Features must be a 3-D temporal array.")
        if labels.ndim != 1 or len(values) != len(labels):
            raise ValueError("Targets must be 1-D and aligned with features.")
        if not np.issubdtype(values.dtype, np.number) or not np.issubdtype(labels.dtype, np.number):
            raise ValueError("Features and targets must be numeric.")
        if not np.isfinite(values).all() or not np.isfinite(labels).all():
            raise ValueError("Features and targets must contain only finite values.")
        self.features = torch.as_tensor(values, dtype=torch.float32)
        self.targets = torch.as_tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.targets[index]
