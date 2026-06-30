from __future__ import annotations

import csv
from pathlib import Path

import torch

from .losses import lejepa_loss
from .models import LeJEPAModel
from .sigreg import SIGReg


def train_lejepa_smoke(
    out_dir: str | Path,
    epochs: int = 2,
    input_dim: int = 64,
    embedding_dim: int = 32,
    lambda_sigreg: float = 0.05,
    seed: int = 42,
) -> dict[str, float | bool]:
    torch.manual_seed(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = LeJEPAModel(input_dim=input_dim, embedding_dim=embedding_dim)
    sigreg = SIGReg(embedding_dim=embedding_dim, seed=seed)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    rows = []
    for epoch in range(epochs):
        base = torch.randn(64, input_dim)
        view_a = base + 0.05 * torch.randn_like(base)
        view_b = base + 0.05 * torch.randn_like(base)
        out = model(view_a, view_b)
        loss, logs = lejepa_loss(out["pred_b"], out["z_b"], torch.cat([out["z_a"], out["z_b"]], dim=0), sigreg, lambda_sigreg)
        opt.zero_grad()
        loss.backward()
        opt.step()
        logs["epoch"] = epoch + 1
        rows.append(logs)
    with (out_dir / "pretrain_log.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[-1].keys()))
        writer.writeheader()
        writer.writerows(rows)
    torch.save({"model": model.state_dict(), "config": {"input_dim": input_dim, "embedding_dim": embedding_dim}}, out_dir / "lejepa_smoke.pt")
    return rows[-1]
