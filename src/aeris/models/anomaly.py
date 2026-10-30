"""Compact sequence autoencoder for telemetry anomaly scoring."""

from __future__ import annotations

import torch
from torch import nn


class SequenceAutoencoder(nn.Module):
    """LSTM sequence autoencoder returning a reconstruction per timestep."""

    def __init__(self, input_size: int = 15, hidden_size: int = 32) -> None:
        super().__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.encoder(x)
        decoded, _ = self.decoder(encoded)
        return self.output(decoded)
