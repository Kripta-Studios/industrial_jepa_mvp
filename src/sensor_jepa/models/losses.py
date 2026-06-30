from __future__ import annotations

import torch
import torch.nn.functional as F


def normalized_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target, dim=-1)
    return F.mse_loss(pred, target)


def variance_regularization(z: torch.Tensor, target_std: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
    z = z - z.mean(dim=0, keepdim=True)
    std = torch.sqrt(z.var(dim=0, unbiased=False) + eps)
    return torch.relu(target_std - std).mean()


def covariance_regularization(z: torch.Tensor) -> torch.Tensor:
    z = z - z.mean(dim=0, keepdim=True)
    n, d = z.shape
    if n < 2:
        return z.new_tensor(0.0)
    cov = (z.T @ z) / (n - 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    return (off_diag.pow(2).sum() / d).float()
