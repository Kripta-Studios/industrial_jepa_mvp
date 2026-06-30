from __future__ import annotations

import torch
from torch import nn


class SmallConvImageEncoder(nn.Module):
    def __init__(self, embedding_dim: int = 128, hidden_dim: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
        )
        self.proj = nn.Linear(hidden_dim, embedding_dim)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fmap = self.forward_features(x)
        pooled = fmap.mean(dim=(2, 3))
        return self.proj(pooled)


def build_visual_encoder(kind: str, embedding_dim: int, hidden_dim: int) -> nn.Module:
    if kind == "small_conv":
        return SmallConvImageEncoder(embedding_dim=embedding_dim, hidden_dim=hidden_dim)
    raise ValueError(f"Unknown visual encoder: {kind}")

