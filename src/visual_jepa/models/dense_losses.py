from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_latent_loss(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
    pred_n = F.normalize(pred, dim=-1)
    target_n = F.normalize(target.detach(), dim=-1)
    per = 1.0 - (pred_n * target_n).sum(dim=-1)
    if valid is not None:
        weights = valid.float()
        return (per * weights).sum() / weights.sum().clamp_min(1.0)
    return per.mean()


def visible_latent_loss(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
    return masked_latent_loss(pred, target, valid)


def deep_supervision_loss(
    pred_states: list[torch.Tensor],
    target_states: list[torch.Tensor],
    target_indices: torch.Tensor,
    valid: torch.Tensor,
    gather_fn,
) -> torch.Tensor:
    if not pred_states or not target_states:
        return pred_states[0].new_tensor(0.0) if pred_states else torch.tensor(0.0)
    losses = []
    for pred, target in zip(pred_states, target_states):
        losses.append(masked_latent_loss(pred, gather_fn(target, target_indices), valid))
    return torch.stack(losses).mean() if losses else target_states[-1].new_tensor(0.0)


def variance_regularization(z: torch.Tensor, target_std: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
    flat = z.reshape(-1, z.shape[-1])
    std = torch.sqrt(flat.var(dim=0, unbiased=False) + eps)
    return torch.relu(target_std - std).mean()


def covariance_regularization(z: torch.Tensor) -> torch.Tensor:
    flat = z.reshape(-1, z.shape[-1])
    flat = flat - flat.mean(dim=0, keepdim=True)
    denom = max(flat.shape[0] - 1, 1)
    cov = flat.T @ flat / denom
    off_diag = cov - torch.diag(torch.diag(cov))
    return off_diag.pow(2).sum() / z.shape[-1]


def collapse_metrics(z: torch.Tensor) -> dict[str, torch.Tensor]:
    flat = z.detach().reshape(-1, z.shape[-1])
    std = flat.std(dim=0, unbiased=False)
    return {
        "embedding_std": std.mean(),
        "embedding_mean": flat.mean().abs(),
        "collapse_score": (std < 0.05).float().mean(),
    }
