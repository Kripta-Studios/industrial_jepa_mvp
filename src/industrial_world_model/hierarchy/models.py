from __future__ import annotations

import torch
import torch.nn as nn


class SimpleFusionState(nn.Module):
    def __init__(self, image_dim: int, sensor_dim: int, action_dim: int, fused_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(image_dim + sensor_dim + action_dim, fused_dim),
            nn.GELU(),
            nn.Linear(fused_dim, fused_dim),
        )

    def forward(self, image_z: torch.Tensor, sensor_z: torch.Tensor, action_z: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([image_z, sensor_z, action_z], dim=-1))
