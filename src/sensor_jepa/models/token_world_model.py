from __future__ import annotations

import torch
from torch import nn


class TokenWorldModel(nn.Module):
    """Predict future dense sensor tokens from current tokens and context/action."""

    def __init__(self, embedding_dim: int, action_dim: int = 0, hidden_dim: int = 256):
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.action_dim = int(action_dim)
        cond_dim = self.action_dim + 1
        self.cond = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.predictor = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, tokens: torch.Tensor, action: torch.Tensor | None = None, horizon: int | torch.Tensor = 1) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"Expected tokens [B,N,D], got {tuple(tokens.shape)}")
        b, n, _ = tokens.shape
        if self.action_dim:
            if action is None:
                action = torch.zeros(b, self.action_dim, device=tokens.device, dtype=tokens.dtype)
            else:
                action = action.to(device=tokens.device, dtype=tokens.dtype)
        else:
            action = torch.empty(b, 0, device=tokens.device, dtype=tokens.dtype)
        if isinstance(horizon, int):
            h = torch.full((b, 1), float(horizon), device=tokens.device, dtype=tokens.dtype)
        else:
            h = horizon.reshape(b, 1).to(device=tokens.device, dtype=tokens.dtype)
        cond = self.cond(torch.cat([action, h], dim=-1)).unsqueeze(1).expand(b, n, -1)
        return self.predictor(torch.cat([tokens, cond], dim=-1))


def token_surprise(pred_tokens: torch.Tensor, target_tokens: torch.Tensor) -> torch.Tensor:
    if pred_tokens.shape != target_tokens.shape:
        raise ValueError("pred_tokens and target_tokens must have the same shape")
    return (pred_tokens - target_tokens.detach()).pow(2).mean(dim=-1)
