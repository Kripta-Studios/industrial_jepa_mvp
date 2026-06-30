from __future__ import annotations

import torch
import torch.nn.functional as F

from .sigreg import SIGReg, collapse_diagnostics


def prediction_loss(pred: torch.Tensor, target: torch.Tensor, mode: str = "mse") -> torch.Tensor:
    if mode == "cosine":
        return 1.0 - F.cosine_similarity(pred, target, dim=-1).mean()
    return F.mse_loss(pred, target)


def lejepa_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    embeddings: torch.Tensor,
    sigreg: SIGReg,
    lambda_sigreg: float = 0.05,
    mode: str = "mse",
) -> tuple[torch.Tensor, dict[str, float | bool]]:
    pred_loss = prediction_loss(pred, target, mode=mode)
    sig = sigreg(embeddings)
    total = pred_loss + float(lambda_sigreg) * sig
    logs = {
        "prediction_loss": float(pred_loss.detach().cpu()),
        "sigreg_loss": float(sig.detach().cpu()),
        "loss": float(total.detach().cpu()),
    }
    logs.update(collapse_diagnostics(embeddings))
    return total, logs
