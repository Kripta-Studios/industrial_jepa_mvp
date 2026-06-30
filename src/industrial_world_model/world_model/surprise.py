from __future__ import annotations

import torch
import torch.nn.functional as F


def latent_surprise(z_hat: torch.Tensor, z_target: torch.Tensor, mode: str = "mse") -> torch.Tensor:
    if mode == "cosine":
        return 1.0 - F.cosine_similarity(z_hat, z_target, dim=-1)
    if mode == "mahalanobis":
        diff = z_hat - z_target
        var = z_target.var(dim=0, unbiased=False).clamp_min(1e-6)
        return (diff.pow(2) / var).mean(dim=-1).sqrt()
    return (z_hat - z_target).pow(2).mean(dim=-1)


def normalize_surprise_by_group(scores: torch.Tensor, groups: list[str]) -> torch.Tensor:
    out = scores.clone().float()
    for g in sorted(set(groups)):
        idx = torch.tensor([i for i, v in enumerate(groups) if v == g], device=out.device)
        vals = out[idx]
        out[idx] = (vals - vals.mean()) / vals.std().clamp_min(1e-6)
    return out


def temporal_surprise_profile(scores: torch.Tensor, top_k: int = 5, alpha: float = 0.3) -> dict[str, torch.Tensor]:
    """Aggregate window/token surprise into deployment-oriented risk signals.

    Returns scalar tensors for one trajectory. The caller can apply this per
    tool, cycle, lot, or line depending on available metadata.
    """
    values = scores.float().reshape(-1)
    if values.numel() == 0:
        zero = torch.tensor(0.0, device=scores.device)
        return {"mean": zero, "max": zero, "topk_mean": zero, "ewma_last": zero}
    k = max(1, min(int(top_k), values.numel()))
    ewma = values[0]
    for value in values[1:]:
        ewma = float(alpha) * value + (1.0 - float(alpha)) * ewma
    return {
        "mean": values.mean(),
        "max": values.max(),
        "topk_mean": values.topk(k).values.mean(),
        "ewma_last": ewma,
    }


def residual_surprise(scores: torch.Tensor, baseline_scores: torch.Tensor) -> torch.Tensor:
    """Residual surprise after a baseline risk signal.

    This is a conservative diagnostic: if residual surprise has no predictive
    value, the world model is not adding value beyond the baseline.
    """
    s = scores.float()
    b = baseline_scores.float().to(device=s.device)
    b = (b - b.mean()) / b.std(unbiased=False).clamp_min(1e-6)
    s_norm = (s - s.mean()) / s.std(unbiased=False).clamp_min(1e-6)
    return s_norm - b
