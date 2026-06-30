from __future__ import annotations

import torch
import torch.nn as nn


class MLPEncoder(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.flatten(1))


class LeJEPAModel(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.encoder = MLPEncoder(input_dim, embedding_dim, hidden_dim)
        self.predictor = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, view_a: torch.Tensor, view_b: torch.Tensor) -> dict[str, torch.Tensor]:
        z_a = self.encoder(view_a)
        z_b = self.encoder(view_b)
        pred_b = self.predictor(z_a)
        return {"z_a": z_a, "z_b": z_b, "pred_b": pred_b}
