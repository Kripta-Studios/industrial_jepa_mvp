from __future__ import annotations

import torch
from torch import nn

from sensor_jepa.models.predictors import MLPPredictor
from visual_jepa.data.patching import random_block_mask_images

from .encoders import build_visual_encoder
from .losses import normalized_mse, variance_regularization


class VisualJEPA(nn.Module):
    def __init__(
        self,
        encoder: str = "small_conv",
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        predictor_hidden_dim: int = 256,
        patch_size: int = 16,
        mask_ratio: float = 0.5,
        sigreg_weight: float = 0.05,
    ):
        super().__init__()
        self.encoder = build_visual_encoder(encoder, embedding_dim, hidden_dim)
        self.predictor = MLPPredictor(embedding_dim, predictor_hidden_dim)
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.sigreg_weight = sigreg_weight

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        mask = random_block_mask_images(x, patch_size=self.patch_size, mask_ratio=self.mask_ratio)
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

    @torch.no_grad()
    def feature_map(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder.forward_features(x)
