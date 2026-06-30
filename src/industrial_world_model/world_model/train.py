from __future__ import annotations

import csv
from pathlib import Path

import torch
import torch.nn.functional as F

from industrial_world_model.lejepa.sigreg import SIGReg, collapse_diagnostics

from .models import LatentWorldModel
from .surprise import latent_surprise, temporal_surprise_profile


def train_world_model_smoke(
    out_dir: str | Path,
    epochs: int = 2,
    latent_dim: int = 32,
    action_dim: int = 8,
    lambda_sigreg: float = 0.05,
    seed: int = 42,
) -> dict[str, float]:
    torch.manual_seed(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = LatentWorldModel(latent_dim=latent_dim, action_dim=action_dim)
    sigreg = SIGReg(embedding_dim=latent_dim, seed=seed)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    rows = []
    for epoch in range(epochs):
        z = torch.randn(128, latent_dim)
        a = torch.randn(128, action_dim)
        z_next = z + 0.2 * torch.tanh(a[:, :1]) + 0.05 * torch.randn_like(z)
        pred = model(z, a, horizon=1)
        prediction = F.mse_loss(pred, z_next)
        sig = sigreg(torch.cat([z, z_next, pred], dim=0))
        loss = prediction + float(lambda_sigreg) * sig
        opt.zero_grad()
        loss.backward()
        opt.step()
        rows.append(
            {
                "epoch": epoch + 1,
                "prediction_loss": float(prediction.detach()),
                "sigreg_loss": float(sig.detach()),
                "loss": float(loss.detach()),
            }
        )
    pred = model(z, a, horizon=1)
    surprise = latent_surprise(pred, z_next)
    profile = temporal_surprise_profile(surprise, top_k=10)
    rows[-1]["surprise_mean"] = float(profile["mean"].detach())
    rows[-1]["surprise_max"] = float(profile["max"].detach())
    rows[-1]["surprise_topk_mean"] = float(profile["topk_mean"].detach())
    rows[-1]["surprise_ewma_last"] = float(profile["ewma_last"].detach())
    rows[-1].update(collapse_diagnostics(pred))
    with (out_dir / "results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[-1].keys()))
        writer.writeheader()
        writer.writerows(rows)
    torch.save({"model": model.state_dict(), "latent_dim": latent_dim, "action_dim": action_dim}, out_dir / "leworldmodel_smoke.pt")
    return rows[-1]
