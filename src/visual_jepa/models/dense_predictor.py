from __future__ import annotations

import torch
from torch import nn


def gather_tokens(tokens: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather token rows from [B,N,D] using [B,K] indices."""
    if tokens.ndim != 3 or indices.ndim != 2:
        raise ValueError("Expected tokens [B,N,D] and indices [B,K]")
    b, _, d = tokens.shape
    idx = indices.unsqueeze(-1).expand(b, indices.shape[1], d)
    return torch.gather(tokens, dim=1, index=idx)


class MLPDensePredictor(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int | None = None):
        super().__init__()
        hidden = int(hidden_dim or embedding_dim * 2)
        self.pos_proj = nn.Linear(embedding_dim, embedding_dim)
        self.net = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, embedding_dim),
        )

    def forward(
        self,
        context_tokens: torch.Tensor,
        target_pos: torch.Tensor,
        context_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if context_valid is None:
            pooled = context_tokens.mean(dim=1, keepdim=True)
        else:
            weights = context_valid.float().unsqueeze(-1)
            pooled = (context_tokens * weights).sum(dim=1, keepdim=True) / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = pooled.expand(-1, target_pos.shape[1], -1)
        return self.net(torch.cat([pooled, self.pos_proj(target_pos)], dim=-1))


class TransformerDensePredictor(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        depth: int = 2,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.query_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        self.pos_proj = nn.Linear(embedding_dim, embedding_dim)
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=embedding_dim,
                    nhead=num_heads,
                    dim_feedforward=int(embedding_dim * mlp_ratio),
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embedding_dim)
        nn.init.trunc_normal_(self.query_token, std=0.02)

    def forward(
        self,
        context_tokens: torch.Tensor,
        target_pos: torch.Tensor,
        context_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, nt, d = target_pos.shape
        target_queries = self.query_token.expand(b, nt, d) + self.pos_proj(target_pos)
        seq = torch.cat([context_tokens, target_queries], dim=1)
        key_padding_mask = None
        if context_valid is not None:
            target_valid = torch.zeros((b, nt), dtype=torch.bool, device=context_tokens.device)
            key_padding_mask = torch.cat([~context_valid.bool(), target_valid], dim=1)
        for block in self.blocks:
            seq = block(seq, src_key_padding_mask=key_padding_mask)
        seq = self.norm(seq)
        return seq[:, -nt:]


def build_dense_predictor(
    kind: str,
    embedding_dim: int,
    hidden_dim: int | None = None,
    depth: int = 2,
    num_heads: int = 3,
) -> nn.Module:
    if kind == "mlp":
        return MLPDensePredictor(embedding_dim=embedding_dim, hidden_dim=hidden_dim)
    if kind == "transformer":
        return TransformerDensePredictor(embedding_dim=embedding_dim, depth=depth, num_heads=num_heads)
    raise ValueError(f"Unknown dense predictor: {kind}")
