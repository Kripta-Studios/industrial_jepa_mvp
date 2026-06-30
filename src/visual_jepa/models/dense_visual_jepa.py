from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from visual_jepa.data.block_masks import BlockMaskGenerator
from visual_jepa.data.dense_patching import patch_mask_to_image

from .dense_losses import collapse_metrics, covariance_regularization, masked_latent_loss, variance_regularization, visible_latent_loss
from .dense_predictor import build_dense_predictor, gather_tokens
from .dense_vit import DenseEncoderOutput, build_dense_vit_encoder
from .ema import clone_as_ema_target, update_ema


class DenseVisualJEPA(nn.Module):
    def __init__(
        self,
        image_size: int = 224,
        channels: int = 3,
        encoder: str = "vit_tiny",
        patch_size: int = 16,
        embedding_dim: int = 192,
        depth: int = 6,
        num_heads: int = 3,
        target_mode: str = "ema",
        ema_momentum: float = 0.996,
        predictor: str = "transformer",
        predictor_hidden_dim: int | None = None,
        predictor_depth: int = 2,
        predictor_heads: int = 3,
        mask_ratio: float = 0.6,
        num_target_blocks: int = 4,
        min_block_size: int = 2,
        max_block_size: int = 6,
        context_min_ratio: float = 0.25,
        target_loss_weight: float = 1.0,
        visible_loss_weight: float = 0.5,
        deep_supervision_weight: float = 0.0,
        variance_weight: float = 0.05,
        covariance_weight: float = 0.0,
        seed: int | None = None,
    ):
        super().__init__()
        self.context_encoder = build_dense_vit_encoder(
            encoder,
            image_size=image_size,
            patch_size=patch_size,
            in_channels=channels,
            embedding_dim=embedding_dim,
            depth=depth,
            num_heads=num_heads,
        )
        self.target_mode = target_mode
        if target_mode == "ema":
            self.target_encoder = clone_as_ema_target(self.context_encoder)
        elif target_mode == "shared":
            self.target_encoder = self.context_encoder
        else:
            raise ValueError("target_mode must be 'ema' or 'shared'")
        self.ema_momentum = float(ema_momentum)
        self.predictor = build_dense_predictor(
            predictor,
            embedding_dim=embedding_dim,
            hidden_dim=predictor_hidden_dim,
            depth=predictor_depth,
            num_heads=predictor_heads,
        )
        self.position_proj = nn.Sequential(nn.Linear(2, embedding_dim), nn.GELU(), nn.Linear(embedding_dim, embedding_dim))
        self.mask_generator = BlockMaskGenerator(
            mask_ratio=mask_ratio,
            num_target_blocks=num_target_blocks,
            min_block_size=min_block_size,
            max_block_size=max_block_size,
            context_min_ratio=context_min_ratio,
            seed=seed,
        )
        self.patch_size = int(patch_size)
        self.target_loss_weight = float(target_loss_weight)
        self.visible_loss_weight = float(visible_loss_weight)
        self.deep_supervision_weight = float(deep_supervision_weight)
        self.variance_weight = float(variance_weight)
        self.covariance_weight = float(covariance_weight)

    @torch.no_grad()
    def update_target_encoder(self) -> None:
        if self.target_mode == "ema":
            update_ema(self.context_encoder, self.target_encoder, self.ema_momentum)

    def _encode_target(self, x: torch.Tensor) -> DenseEncoderOutput:
        with torch.no_grad():
            out = self.target_encoder.forward_tokens(x)
        return out

    def _make_context_image(self, x: torch.Tensor, target_mask: torch.Tensor, grid_shape: tuple[int, int]) -> torch.Tensor:
        image_mask = patch_mask_to_image(target_mask, grid_shape, self.patch_size)
        image_mask = image_mask[:, :, : x.shape[-2], : x.shape[-1]]
        return x.masked_fill(image_mask.bool(), 0.0)

    def forward(self, x: torch.Tensor) -> dict[str, Any]:
        # First target pass gives the grid; the target branch is stop-gradient.
        target_out = self._encode_target(x)
        masks = self.mask_generator(x.shape[0], target_out.grid_shape, device=x.device)
        context_img = self._make_context_image(x, masks.target_mask, target_out.grid_shape)
        context_out = self.context_encoder.forward_tokens(context_img)

        pos = self.position_proj(target_out.positions).unsqueeze(0).expand(x.shape[0], -1, -1)
        context_tokens = gather_tokens(context_out.tokens, masks.visible_indices)
        context_valid = masks.visible_valid
        target_pos = gather_tokens(pos, masks.target_indices)
        pred_target = self.predictor(context_tokens, target_pos, context_valid=context_valid)
        target_tokens = gather_tokens(target_out.tokens, masks.target_indices)

        masked_loss = masked_latent_loss(pred_target, target_tokens, masks.target_valid)
        loss = self.target_loss_weight * masked_loss
        visible_loss = x.new_tensor(0.0)
        if self.visible_loss_weight > 0:
            visible_pos = gather_tokens(pos, masks.visible_indices)
            pred_visible = self.predictor(context_tokens, visible_pos, context_valid=context_valid)
            target_visible = gather_tokens(target_out.tokens, masks.visible_indices)
            visible_loss = visible_latent_loss(pred_visible, target_visible, masks.visible_valid)
            loss = loss + self.visible_loss_weight * visible_loss

        deep_loss = x.new_tensor(0.0)
        if self.deep_supervision_weight > 0 and target_out.hidden_states:
            hidden_losses = []
            for hidden in target_out.hidden_states[:-1]:
                hidden_losses.append(masked_latent_loss(pred_target, gather_tokens(hidden, masks.target_indices), masks.target_valid))
            if hidden_losses:
                deep_loss = torch.stack(hidden_losses).mean()
                loss = loss + self.deep_supervision_weight * deep_loss

        var_loss = variance_regularization(torch.cat([context_out.tokens, target_out.tokens], dim=1))
        cov_loss = covariance_regularization(context_out.tokens) if self.covariance_weight > 0 else x.new_tensor(0.0)
        loss = loss + self.variance_weight * var_loss + self.covariance_weight * cov_loss
        metrics = collapse_metrics(target_out.tokens)
        return {
            "loss": loss,
            "masked_loss": masked_loss.detach(),
            "visible_loss": visible_loss.detach(),
            "deep_loss": deep_loss.detach(),
            "variance_loss": var_loss.detach(),
            "covariance_loss": cov_loss.detach(),
            "embedding_std": metrics["embedding_std"],
            "embedding_mean": metrics["embedding_mean"],
            "collapse_score": metrics["collapse_score"],
            "pred_target": pred_target,
            "target_tokens": target_tokens,
            "target_valid": masks.target_valid,
            "target_mask": masks.target_mask,
            "target_indices": masks.target_indices,
            "visible_mask": masks.visible_mask,
            "grid_shape": target_out.grid_shape,
        }

    @torch.no_grad()
    def encode_dense(self, x: torch.Tensor) -> dict[str, Any]:
        out = self.context_encoder.forward_tokens(x)
        return {"tokens": F.normalize(out.tokens, dim=-1), "grid_shape": out.grid_shape, "hidden_states": out.hidden_states}

    @torch.no_grad()
    def latent_error_map(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        out = self.forward(x)
        b, n = out["target_mask"].shape
        scores = torch.zeros((b, n), dtype=out["pred_target"].dtype, device=x.device)
        pred = F.normalize(out["pred_target"], dim=-1)
        target = F.normalize(out["target_tokens"], dim=-1)
        per = 1.0 - (pred * target).sum(dim=-1)
        scores.scatter_(1, out["target_indices"], per * out["target_valid"].float())
        return scores.reshape(b, 1, out["grid_shape"][0], out["grid_shape"][1]), out["grid_shape"]


def build_dense_visual_jepa_from_config(cfg: dict[str, Any]) -> DenseVisualJEPA:
    model_cfg = cfg.get("model", {})
    masking_cfg = cfg.get("masking", {})
    loss_cfg = cfg.get("loss", {})
    dataset_cfg = cfg.get("dataset", {})
    return DenseVisualJEPA(
        image_size=int(dataset_cfg.get("image_size", model_cfg.get("image_size", 224))),
        channels=int(dataset_cfg.get("channels", 3)),
        encoder=model_cfg.get("encoder", "vit_tiny"),
        patch_size=int(model_cfg.get("patch_size", 16)),
        embedding_dim=int(model_cfg.get("embedding_dim", 192)),
        depth=int(model_cfg.get("depth", 6)),
        num_heads=int(model_cfg.get("num_heads", 3)),
        target_mode=model_cfg.get("target_mode", "ema"),
        ema_momentum=float(model_cfg.get("ema_momentum", 0.996)),
        predictor=model_cfg.get("predictor", "transformer"),
        predictor_hidden_dim=model_cfg.get("predictor_hidden_dim"),
        predictor_depth=int(model_cfg.get("predictor_depth", 2)),
        predictor_heads=int(model_cfg.get("predictor_heads", model_cfg.get("num_heads", 3))),
        mask_ratio=float(masking_cfg.get("mask_ratio", 0.6)),
        num_target_blocks=int(masking_cfg.get("num_target_blocks", 4)),
        min_block_size=int(masking_cfg.get("min_block_size", 2)),
        max_block_size=int(masking_cfg.get("max_block_size", 6)),
        context_min_ratio=float(masking_cfg.get("context_min_ratio", 0.25)),
        target_loss_weight=float(loss_cfg.get("target_loss_weight", 1.0)),
        visible_loss_weight=float(loss_cfg.get("visible_loss_weight", 0.5)),
        deep_supervision_weight=float(loss_cfg.get("deep_supervision_weight", 0.0)),
        variance_weight=float(loss_cfg.get("variance_weight", 0.05)),
        covariance_weight=float(loss_cfg.get("covariance_weight", 0.0)),
        seed=cfg.get("seed"),
    )
