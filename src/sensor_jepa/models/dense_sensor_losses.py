from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class TokenSIGReg(nn.Module):
    """Projection-based isotropic regularizer for token embeddings."""

    def __init__(self, embedding_dim: int, num_projections: int = 128, seed: int = 42, eps: float = 1e-6):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        projection = torch.randn(embedding_dim, num_projections, generator=gen)
        projection = F.normalize(projection, dim=0)
        self.register_buffer("projection", projection)
        self.eps = float(eps)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        flat = tokens.reshape(-1, tokens.shape[-1])
        y = flat @ self.projection.to(device=flat.device, dtype=flat.dtype)
        mean_loss = y.mean(dim=0).pow(2).mean()
        std = y.std(dim=0, unbiased=False).clamp_min(self.eps)
        return mean_loss + (std - 1.0).pow(2).mean()


def masked_latent_token_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target.detach(), dim=-1)
    return (1.0 - (pred * target).sum(dim=-1)).mean()


def visible_latent_token_loss(context_tokens: torch.Tensor, target_tokens: torch.Tensor, context_mask: torch.Tensor) -> torch.Tensor:
    if context_mask.ndim == 1:
        context_mask = context_mask.unsqueeze(0).expand(context_tokens.shape[0], -1)
    if not bool(context_mask.any()):
        return context_tokens.new_tensor(0.0)
    pred = F.normalize(context_tokens[context_mask], dim=-1)
    target = F.normalize(target_tokens.detach()[context_mask], dim=-1)
    return (1.0 - (pred * target).sum(dim=-1)).mean()


def future_token_latent_loss(pred: torch.Tensor, future_target: torch.Tensor) -> torch.Tensor:
    return masked_latent_token_loss(pred, future_target)


def token_variance_regularization(tokens: torch.Tensor, target_std: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
    flat = tokens.reshape(-1, tokens.shape[-1])
    flat = flat - flat.mean(dim=0, keepdim=True)
    std = torch.sqrt(flat.var(dim=0, unbiased=False) + eps)
    return torch.relu(target_std - std).mean()


def token_covariance_regularization(tokens: torch.Tensor) -> torch.Tensor:
    flat = tokens.reshape(-1, tokens.shape[-1])
    if flat.shape[0] < 2:
        return flat.new_tensor(0.0)
    flat = flat - flat.mean(dim=0, keepdim=True)
    cov = (flat.T @ flat) / (flat.shape[0] - 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    return off_diag.pow(2).sum() / flat.shape[-1]


def collapse_metrics(tokens: torch.Tensor) -> dict[str, torch.Tensor]:
    flat = tokens.detach().reshape(-1, tokens.shape[-1])
    std = flat.std(dim=0, unbiased=False)
    centered = flat - flat.mean(dim=0, keepdim=True)
    cov = centered.T @ centered / max(centered.shape[0] - 1, 1)
    eigvals = torch.linalg.eigvalsh(cov).clamp_min(1e-12)
    probs = eigvals / eigvals.sum().clamp_min(1e-12)
    effective_rank = torch.exp(-(probs * probs.log()).sum())
    return {
        "embedding_mean": flat.mean(),
        "embedding_std": std.mean(),
        "collapse_score": (std < 1e-3).float().mean(),
        "effective_rank_ratio": effective_rank / max(tokens.shape[-1], 1),
    }


def token_prediction_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target.detach(), dim=-1)
    return 1.0 - (pred * target).sum(dim=-1)
