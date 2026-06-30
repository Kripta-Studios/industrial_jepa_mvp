from __future__ import annotations

import torch
from torch import nn


def masked_context_mean(tokens: torch.Tensor, context_mask: torch.Tensor | None) -> torch.Tensor:
    if context_mask is None:
        return tokens.mean(dim=1)
    if context_mask.ndim == 1:
        context_mask = context_mask.unsqueeze(0).expand(tokens.shape[0], -1)
    weights = context_mask.to(tokens.dtype).unsqueeze(-1)
    denom = weights.sum(dim=1).clamp_min(1.0)
    return (tokens * weights).sum(dim=1) / denom


class MLPDenseSensorPredictor(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, context_tokens: torch.Tensor, context_mask: torch.Tensor, target_pos_embed: torch.Tensor) -> torch.Tensor:
        summary = masked_context_mean(context_tokens, context_mask).unsqueeze(1).expand(-1, target_pos_embed.shape[1], -1)
        return self.net(torch.cat([summary, target_pos_embed], dim=-1))


class TransformerDenseSensorPredictor(nn.Module):
    def __init__(self, embedding_dim: int, depth: int = 2, num_heads: int = 4):
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        self.summary_proj = nn.Linear(embedding_dim, embedding_dim)
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=embedding_dim,
                    nhead=num_heads,
                    dim_feedforward=embedding_dim * 4,
                    dropout=0.0,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embedding_dim)
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(self, context_tokens: torch.Tensor, context_mask: torch.Tensor, target_pos_embed: torch.Tensor) -> torch.Tensor:
        summary = self.summary_proj(masked_context_mean(context_tokens, context_mask)).unsqueeze(1)
        query = self.mask_token + target_pos_embed + summary
        h = query
        for layer in self.layers:
            h = layer(h)
        return self.norm(h)


def build_dense_sensor_predictor(kind: str, embedding_dim: int, hidden_dim: int, depth: int, num_heads: int) -> nn.Module:
    if kind == "mlp":
        return MLPDenseSensorPredictor(embedding_dim=embedding_dim, hidden_dim=hidden_dim)
    if kind == "transformer":
        return TransformerDenseSensorPredictor(embedding_dim=embedding_dim, depth=depth, num_heads=num_heads)
    raise ValueError(f"Unknown dense sensor predictor: {kind}")
