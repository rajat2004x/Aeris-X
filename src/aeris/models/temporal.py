"""Temporal deep-learning models for sequence-to-one RUL regression."""

from __future__ import annotations

import math

import torch
from torch import nn


class LSTMRegressor(nn.Module):
    """LSTM that predicts one RUL value from a temporal window."""

    def __init__(
        self,
        input_size: int = 15,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        return self.head(self.dropout(output[:, -1, :])).squeeze(-1)


class GRURegressor(nn.Module):
    """GRU that predicts one RUL value from a temporal window."""

    def __init__(
        self,
        input_size: int = 15,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.gru(x)
        return self.head(self.dropout(output[:, -1, :])).squeeze(-1)


class TemporalBlock(nn.Module):
    """Residual dilated convolutional block with sequence-length preservation."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(input_channels, output_channels, kernel_size, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(output_channels, output_channels, kernel_size, padding=padding, dilation=dilation)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Conv1d(input_channels, output_channels, 1) if input_channels != output_channels else nn.Identity()

    @staticmethod
    def _trim(sequence: torch.Tensor, padding: int) -> torch.Tensor:
        return sequence[:, :, :-padding] if padding else sequence

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        padding = self.conv1.padding[0]
        result = self._trim(self.conv1(x), padding)
        result = self.dropout(self.activation(result))
        result = self._trim(self.conv2(result), padding)
        result = self.dropout(self.activation(result))
        return self.activation(result + self.residual(x))


class TCNRegressor(nn.Module):
    """Residual dilated temporal convolutional network for RUL regression."""

    def __init__(
        self,
        input_size: int = 15,
        channels: tuple[int, ...] = (64, 64, 128),
        kernel_size: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        blocks = []
        previous = input_size
        for index, channels_out in enumerate(channels):
            blocks.append(TemporalBlock(previous, channels_out, kernel_size, 2**index, dropout))
            previous = channels_out
        self.network = nn.Sequential(*blocks)
        self.head = nn.Linear(previous, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.network(x.transpose(1, 2))
        return self.head(output[:, :, -1]).squeeze(-1)


class PositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_length: int = 30) -> None:
        super().__init__()
        positions = torch.arange(max_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        encoding = torch.zeros(max_length, d_model)
        encoding[:, 0::2] = torch.sin(positions * div_term)
        encoding[:, 1::2] = torch.cos(positions * div_term)
        self.register_buffer("encoding", encoding.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.encoding[:, : x.size(1)]


class TransformerRegressor(nn.Module):
    """Compact Transformer encoder for temporal RUL regression."""

    def __init__(
        self,
        input_size: int = 15,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        max_length: int = 30,
    ) -> None:
        super().__init__()
        if d_model % nhead:
            raise ValueError("d_model must be divisible by nhead.")
        self.projection = nn.Linear(input_size, d_model)
        self.position = PositionalEncoding(d_model, max_length)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(self.position(self.projection(x)))
        return self.head(encoded[:, -1, :]).squeeze(-1)
