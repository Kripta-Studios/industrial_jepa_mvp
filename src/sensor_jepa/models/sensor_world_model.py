from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .encoders import build_sensor_encoder
from .losses import variance_regularization


class ActionConditionedSensorWorldModel(nn.Module):
    """LeWorldModel-style latent next-state predictor for industrial sensor windows."""

    def __init__(
        self,
        input_channels: int,
        action_dim: int,
        encoder: str = "conv1d",
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        predictor_hidden_dim: int = 256,
        sigreg_weight: float = 0.05,
    ):
        super().__init__()
        self.encoder = build_sensor_encoder(encoder, input_channels, embedding_dim, hidden_dim)
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.predictor = nn.Sequential(
            nn.Linear(embedding_dim * 2, predictor_hidden_dim),
            nn.GELU(),
            nn.Linear(predictor_hidden_dim, predictor_hidden_dim),
            nn.GELU(),
            nn.Linear(predictor_hidden_dim, embedding_dim),
        )
        self.sigreg_weight = sigreg_weight

    def predict_next_embedding(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        a = self.action_encoder(action)
        return self.predictor(torch.cat([z, a], dim=-1))

    def forward(self, x: torch.Tensor, action: torch.Tensor, x_next: torch.Tensor) -> dict[str, torch.Tensor]:
        pred = self.predict_next_embedding(x, action)
        target = self.encoder(x_next)
        pred_n = F.normalize(pred, dim=-1)
        target_n = F.normalize(target, dim=-1)
        per_sample = 1.0 - (pred_n * target_n).sum(dim=-1)
        pred_loss = per_sample.mean()
        sigreg = variance_regularization(torch.cat([pred, target], dim=0))
        loss = pred_loss + self.sigreg_weight * sigreg
        return {
            "loss": loss,
            "pred_loss": pred_loss.detach(),
            "sigreg": sigreg.detach(),
            "embedding_std": target.detach().std(dim=0, unbiased=False).mean(),
            "per_sample_error": per_sample.detach(),
        }

