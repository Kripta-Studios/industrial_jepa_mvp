from __future__ import annotations

import math

import torch
from torch import nn


class Conv1DEncoder(nn.Module):
    def __init__(self, input_channels: int, embedding_dim: int = 128, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_channels, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.proj = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        x = x.transpose(1, 2)
        h = self.net(x)
        return self.proj(h)


class TransformerTimeSeriesEncoder(nn.Module):
    def __init__(
        self,
        input_channels: int,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        max_len: int = 512,
    ):
        super().__init__()
        self.input = nn.Linear(input_channels, hidden_dim)
        self.pos = nn.Parameter(torch.zeros(1, max_len, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.out = nn.Linear(hidden_dim, embedding_dim)
        nn.init.normal_(self.pos, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = x.shape[1]
        h = self.input(x) * math.sqrt(self.input.out_features)
        h = h + self.pos[:, :t]
        h = self.encoder(h)
        return self.out(h.mean(dim=1))


def build_sensor_encoder(kind: str, input_channels: int, embedding_dim: int, hidden_dim: int) -> nn.Module:
    if kind == "transformer":
        return TransformerTimeSeriesEncoder(input_channels, embedding_dim, hidden_dim)
    if kind == "conv1d":
        return Conv1DEncoder(input_channels, embedding_dim, hidden_dim)
    raise ValueError(f"Unknown sensor encoder: {kind}")

