from __future__ import annotations

import torch
from torch import nn

from sensor_jepa.data.temporal_masks import TemporalMaskGenerator

from .dense_sensor_encoder import build_dense_sensor_encoder
from .dense_sensor_losses import (
    TokenSIGReg,
    collapse_metrics,
    masked_latent_token_loss,
    token_covariance_regularization,
    token_prediction_error,
    token_variance_regularization,
    visible_latent_token_loss,
)
from .dense_sensor_predictor import build_dense_sensor_predictor
from .sensor_ema import clone_as_ema_target, update_ema


class DenseSensorJEPA(nn.Module):
    def __init__(
        self,
        input_channels: int,
        encoder: str = "temporal_transformer",
        embedding_dim: int = 256,
        depth: int = 4,
        num_heads: int = 4,
        temporal_patch_size: int = 2,
        temporal_patch_stride: int = 1,
        tokenization_mode: str = "multichannel_token",
        target_mode: str = "ema",
        ema_momentum: float = 0.996,
        predictor: str = "transformer",
        predictor_hidden_dim: int = 512,
        predictor_depth: int = 2,
        temporal_mask_ratio: float = 0.4,
        num_target_spans: int = 2,
        min_span: int = 1,
        max_span: int = 4,
        target_loss_weight: float = 1.0,
        visible_loss_weight: float = 0.5,
        future_loss_weight: float = 0.0,
        variance_weight: float = 0.05,
        covariance_weight: float = 0.0,
        sigreg_weight: float = 0.0,
        sigreg_num_projections: int = 128,
        seed: int = 42,
    ):
        super().__init__()
        self.context_encoder = build_dense_sensor_encoder(
            encoder,
            input_channels=input_channels,
            embedding_dim=embedding_dim,
            depth=depth,
            num_heads=num_heads,
            temporal_patch_size=temporal_patch_size,
            temporal_patch_stride=temporal_patch_stride,
            tokenization_mode=tokenization_mode,
        )
        self.target_mode = target_mode
        if target_mode == "ema":
            self.target_encoder = clone_as_ema_target(self.context_encoder)
        elif target_mode == "shared":
            self.target_encoder = self.context_encoder
        else:
            raise ValueError("target_mode must be 'shared' or 'ema'")
        self.predictor = build_dense_sensor_predictor(
            predictor,
            embedding_dim=embedding_dim,
            hidden_dim=predictor_hidden_dim,
            depth=predictor_depth,
            num_heads=num_heads,
        )
        self.ema_momentum = ema_momentum
        self.temporal_mask_ratio = temporal_mask_ratio
        self.num_target_spans = num_target_spans
        self.min_span = min_span
        self.max_span = max_span
        self.target_loss_weight = target_loss_weight
        self.visible_loss_weight = visible_loss_weight
        self.future_loss_weight = future_loss_weight
        self.variance_weight = variance_weight
        self.covariance_weight = covariance_weight
        self.sigreg_weight = float(sigreg_weight)
        self.sigreg = TokenSIGReg(embedding_dim=embedding_dim, num_projections=sigreg_num_projections, seed=seed)
        self.latent_mask_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        nn.init.normal_(self.latent_mask_token, std=0.02)

    def _target_tokens(self, x: torch.Tensor) -> torch.Tensor:
        if self.target_mode == "ema":
            with torch.no_grad():
                return self.target_encoder(x)["tokens"]  # type: ignore[index]
        return self.context_encoder(x)["tokens"].detach()  # type: ignore[index]

    @staticmethod
    def _gather(tokens: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        if indices.ndim == 1:
            return tokens[:, indices]
        expanded = indices.unsqueeze(-1).expand(-1, -1, tokens.shape[-1])
        return tokens.gather(1, expanded)

    def make_masks(self, num_tokens: int, batch_size: int, device: torch.device, seed: int | None = None) -> dict[str, object]:
        generator = TemporalMaskGenerator(
            num_tokens=num_tokens,
            temporal_mask_ratio=self.temporal_mask_ratio,
            num_target_spans=self.num_target_spans,
            min_span=self.min_span,
            max_span=self.max_span,
            seed=seed,
        )
        return generator(batch_size=batch_size, device=device)

    def forward(
        self,
        x: torch.Tensor,
        x_future: torch.Tensor | None = None,
        masks: dict[str, object] | None = None,
        return_hidden_states: bool = False,
    ) -> dict[str, torch.Tensor]:
        context_out = self.context_encoder(x, return_hidden_states=return_hidden_states)
        context_tokens = context_out["tokens"]  # type: ignore[index]
        target_tokens_full = self._target_tokens(x)
        b, n, _ = context_tokens.shape
        if masks is None:
            masks = self.make_masks(n, b, x.device)
        context_mask = masks["context_mask"]  # type: ignore[index]
        target_indices = masks["target_indices"]  # type: ignore[index]
        if target_indices.ndim == 2 and torch.all(target_indices == target_indices[:1]):
            pos_embed = self.context_encoder.positional_embeddings(target_indices[0]).expand(b, -1, -1)  # type: ignore[attr-defined]
        else:
            pos_embed = self.context_encoder.positional_embeddings(target_indices)  # type: ignore[attr-defined]
        target_tokens = self._gather(target_tokens_full, target_indices)
        predictor_tokens = torch.where(context_mask.unsqueeze(-1), context_tokens, self.latent_mask_token.to(context_tokens.dtype))
        pred_target = self.predictor(predictor_tokens, context_mask, pos_embed)

        target_loss = masked_latent_token_loss(pred_target, target_tokens)
        visible_loss = visible_latent_token_loss(context_tokens, target_tokens_full, context_mask)
        variance_loss = token_variance_regularization(target_tokens_full)
        covariance_loss = token_covariance_regularization(target_tokens_full) if self.covariance_weight else target_loss.new_tensor(0.0)
        sigreg_loss = self.sigreg(torch.cat([context_tokens, target_tokens_full], dim=1)) if self.sigreg_weight else target_loss.new_tensor(0.0)
        future_loss = target_loss.new_tensor(0.0)
        if x_future is not None and self.future_loss_weight:
            future_tokens = self._gather(self._target_tokens(x_future), target_indices)
            future_loss = masked_latent_token_loss(pred_target, future_tokens)
        loss = (
            self.target_loss_weight * target_loss
            + self.visible_loss_weight * visible_loss
            + self.future_loss_weight * future_loss
            + self.variance_weight * variance_loss
            + self.covariance_weight * covariance_loss
            + self.sigreg_weight * sigreg_loss
        )
        metrics = collapse_metrics(target_tokens_full)
        return {
            "loss": loss,
            "target_loss": target_loss.detach(),
            "visible_loss": visible_loss.detach(),
            "future_loss": future_loss.detach(),
            "variance_loss": variance_loss.detach(),
            "covariance_loss": covariance_loss.detach(),
            "sigreg_loss": sigreg_loss.detach(),
            "pred_target": pred_target,
            "target_embeddings": target_tokens.detach(),
            "context_tokens": context_tokens,
            "target_mask": masks["target_mask"],  # type: ignore[index]
            "context_mask": context_mask,
            "token_prediction_error": token_prediction_error(pred_target.detach(), target_tokens.detach()),
            **{k: v.detach() for k, v in metrics.items()},
        }

    @torch.no_grad()
    def update_target_encoder(self) -> None:
        if self.target_mode == "ema":
            update_ema(self.context_encoder, self.target_encoder, self.ema_momentum)

    @torch.no_grad()
    def encode_tokens(self, x: torch.Tensor, batch_size: int | None = None) -> torch.Tensor:
        if batch_size is None:
            return self.context_encoder(x)["tokens"]  # type: ignore[index]
        outs = []
        for i in range(0, len(x), batch_size):
            outs.append(self.context_encoder(x[i : i + batch_size])["tokens"])  # type: ignore[index]
        return torch.cat(outs, dim=0)
