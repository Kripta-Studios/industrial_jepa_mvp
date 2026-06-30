from __future__ import annotations

import torch
from torch import nn

from .encoders import build_sensor_encoder
from .losses import normalized_mse, variance_regularization
from .predictors import MLPPredictor


def random_sensor_mask(
    x: torch.Tensor,
    temporal_mask_ratio: float = 0.4,
    channel_mask_ratio: float = 0.15,
) -> torch.Tensor:
    b, t, c = x.shape
    mask = torch.zeros((b, t, c), dtype=torch.bool, device=x.device)
    n_t = max(1, int(round(t * temporal_mask_ratio)))
    n_c = max(0, int(round(c * channel_mask_ratio)))
    for i in range(b):
        start = torch.randint(0, max(1, t - n_t + 1), (1,), device=x.device).item()
        mask[i, start : start + n_t, :] = True
        if n_c:
            idx = torch.randperm(c, device=x.device)[:n_c]
            mask[i, :, idx] = True
    return mask


class SensorJEPA(nn.Module):
    def __init__(
        self,
        input_channels: int,
        encoder: str = "conv1d",
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        predictor_hidden_dim: int = 256,
        temporal_mask_ratio: float = 0.4,
        channel_mask_ratio: float = 0.15,
        sigreg_weight: float = 0.05,
    ):
        super().__init__()
        self.encoder = build_sensor_encoder(encoder, input_channels, embedding_dim, hidden_dim)
        self.predictor = MLPPredictor(embedding_dim, predictor_hidden_dim)
        self.temporal_mask_ratio = temporal_mask_ratio
        self.channel_mask_ratio = channel_mask_ratio
        self.sigreg_weight = sigreg_weight

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        mask = random_sensor_mask(x, self.temporal_mask_ratio, self.channel_mask_ratio)
        context = x.masked_fill(mask, 0.0)
        context_z = self.encoder(context)
        target_z = self.encoder(x)
        pred_z = self.predictor(context_z)
        pred_loss = normalized_mse(pred_z, target_z)
        sigreg = variance_regularization(target_z)
        loss = pred_loss + self.sigreg_weight * sigreg
        return {
            "loss": loss,
            "pred_loss": pred_loss.detach(),
            "sigreg": sigreg.detach(),
            "embedding_std": target_z.detach().std(dim=0, unbiased=False).mean(),
        }

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
