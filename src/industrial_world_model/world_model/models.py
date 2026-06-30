from __future__ import annotations

import torch
import torch.nn as nn


class LatentWorldModel(nn.Module):
    def __init__(self, latent_dim: int = 32, action_dim: int = 8, hidden_dim: int = 128):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim + action_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z_t: torch.Tensor, action_t: torch.Tensor | None = None, horizon: int | torch.Tensor = 1) -> torch.Tensor:
        if action_t is None:
            action_t = torch.zeros(z_t.shape[0], self.action_dim, device=z_t.device, dtype=z_t.dtype)
        if isinstance(horizon, int):
            h = torch.full((z_t.shape[0], 1), float(horizon), device=z_t.device, dtype=z_t.dtype)
        else:
            h = horizon.reshape(-1, 1).to(device=z_t.device, dtype=z_t.dtype)
        return self.predictor(torch.cat([z_t, action_t, h], dim=-1))
