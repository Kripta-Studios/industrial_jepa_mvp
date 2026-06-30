from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SIGReg(nn.Module):
    """Small SIGReg-style isotropic Gaussian regularizer.

    The loss matches random-projected embedding means to zero and variances to
    one. It is deliberately simple and stable for MVP smoke training.
    """

    def __init__(self, embedding_dim: int, num_projections: int = 128, seed: int = 42, eps: float = 1e-6):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        proj = torch.randn(embedding_dim, num_projections, generator=gen)
        proj = torch.nn.functional.normalize(proj, dim=0)
        self.register_buffer("projection", proj)
        self.eps = eps

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim > 2:
            z = z.reshape(-1, z.shape[-1])
        y = z @ self.projection.to(z.device)
        mean_loss = y.mean(dim=0).pow(2).mean()
        std = y.std(dim=0).clamp_min(self.eps)
        var_loss = (std - 1.0).pow(2).mean()
        return mean_loss + var_loss


def collapse_diagnostics(z: torch.Tensor) -> dict[str, float | bool]:
    if z.ndim > 2:
        z = z.reshape(-1, z.shape[-1])
    z = z.detach().float()
    std = z.std(dim=0)
    pairwise = torch.pdist(z[: min(z.shape[0], 256)])
    centered = z - z.mean(dim=0, keepdim=True)
    denom = max(centered.shape[0] - 1, 1)
    cov = centered.T @ centered / denom
    eigvals = torch.linalg.eigvalsh(cov).clamp_min(1e-12)
    probs = eigvals / eigvals.sum().clamp_min(1e-12)
    entropy = -(probs * probs.log()).sum()
    effective_rank = torch.exp(entropy)
    whitened = F.normalize(centered, dim=-1)
    gram = whitened @ whitened.T
    off_diag = gram - torch.diag(torch.diag(gram))
    isotropy_score = float(off_diag.abs().mean().cpu()) if off_diag.numel() else 0.0
    min_var = float(std.min().cpu()) if std.numel() else 0.0
    mean_var = float(std.mean().cpu()) if std.numel() else 0.0
    pair_mean = float(pairwise.mean().cpu()) if pairwise.numel() else 0.0
    pair_std = float(pairwise.std(unbiased=False).cpu()) if pairwise.numel() else 0.0
    collapse_score = float(1.0 / max(mean_var, 1e-6))
    return {
        "embedding_variance_min": min_var,
        "embedding_variance_mean": mean_var,
        "pairwise_distance_mean": pair_mean,
        "pairwise_distance_std": pair_std,
        "effective_rank": float(effective_rank.cpu()),
        "effective_rank_ratio": float((effective_rank / max(z.shape[-1], 1)).cpu()),
        "isotropy_score": isotropy_score,
        "collapse_score": collapse_score,
        "collapse_flag": bool(mean_var < 1e-3 or pair_mean < 1e-3),
    }
