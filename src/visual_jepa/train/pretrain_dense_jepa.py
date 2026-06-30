from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from common.config import get_device_name
from common.paths import ensure_dir
from common.plots import plot_history
from common.reports import write_markdown_report
from common.seed import seed_everything
from visual_jepa.data.industrial import prepare_dense_visual_data
from visual_jepa.models.dense_visual_jepa import DenseVisualJEPA, build_dense_visual_jepa_from_config


def dense_output_root(cfg: dict[str, Any]) -> Path:
    return ensure_dir(cfg.get("eval", {}).get("output_dir", "outputs/visual_jepa/dense_pretrain"))


@torch.no_grad()
def _evaluate_loss(model: DenseVisualJEPA, loader: DataLoader, device: str) -> float:
    if len(loader.dataset) == 0:
        return float("nan")
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        xb = batch["image"].to(device)
        out = model(xb)
        total += float(out["loss"].detach().cpu()) * len(xb)
        n += len(xb)
    return total / max(n, 1)


def pretrain_dense_visual_jepa(cfg: dict[str, Any]) -> tuple[Path, list[dict[str, Any]]]:
    seed_everything(int(cfg.get("seed", 42)))
    device = get_device_name(cfg.get("device", "auto"))
    bundle = prepare_dense_visual_data(cfg)
    model = build_dense_visual_jepa_from_config(cfg).to(device)
    train_cfg = cfg.get("train", {})
    out_root = dense_output_root(cfg)
    ckpt_dir = ensure_dir(out_root / "checkpoints")
    ensure_dir(out_root / "plots")
    with (out_root / "config_used.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    train_loader = DataLoader(
        bundle.train_dataset,
        batch_size=int(train_cfg.get("batch_size", 32)),
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 0)),
    )
    val_loader = DataLoader(
        bundle.val_dataset,
        batch_size=int(train_cfg.get("batch_size", 32)),
        shuffle=False,
        num_workers=int(train_cfg.get("num_workers", 0)),
    )
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(train_cfg.get("lr", 3e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.05)),
    )
    amp = bool(train_cfg.get("amp", True)) and device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    history: list[dict[str, Any]] = []
    best_loss = float("inf")
    best_path = ckpt_dir / "best.pt"
    latest_path = ckpt_dir / "latest.pt"
    start = time.time()
    epochs = int(train_cfg.get("epochs", 20))
    for epoch in range(1, epochs + 1):
        model.train()
        agg = {
            "loss": 0.0,
            "masked_loss": 0.0,
            "visible_loss": 0.0,
            "deep_loss": 0.0,
            "variance_loss": 0.0,
            "embedding_std": 0.0,
            "embedding_mean": 0.0,
            "collapse_score": 0.0,
        }
        n = 0
        for batch in train_loader:
            xb = batch["image"].to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp):
                out = model(xb)
                loss = out["loss"]
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            model.update_target_encoder()
            bs = len(xb)
            n += bs
            for k in agg:
                agg[k] += float(out[k].detach().cpu()) * bs
        row = {"epoch": epoch, **{k: v / max(n, 1) for k, v in agg.items()}}
        row["val_loss"] = _evaluate_loss(model, val_loader, device)
        history.append(row)
        state = {"model_state": model.state_dict(), "cfg": cfg, "history": history, "epoch": epoch}
        torch.save(state, latest_path)
        val_for_best = row["val_loss"] if row["val_loss"] == row["val_loss"] else row["loss"]
        if val_for_best < best_loss:
            best_loss = val_for_best
            torch.save(state, best_path)

    pd.DataFrame(history).to_csv(out_root / "metrics.csv", index=False)
    plot_history(history, out_root / "plots" / "dense_pretrain_loss.png", y_key="loss")
    write_markdown_report(
        out_root / "pretrain_summary.md",
        "DenseVisualJEPA Pretraining",
        {
            "Setup": (
                f"Device: `{device}`\n\n"
                f"Train images: `{len(bundle.train_dataset)}`\n\n"
                f"Val images: `{len(bundle.val_dataset)}`\n\n"
                f"Epochs: `{epochs}`"
            ),
            "Result": (
                f"Final train loss: `{history[-1]['loss']:.6f}`\n\n"
                f"Final val loss: `{history[-1]['val_loss']:.6f}`\n\n"
                f"Elapsed seconds: `{time.time() - start:.2f}`"
            ),
            "Claims": "Pretraining loss alone is not an anomaly-detection claim. Use memory/PaDiM benchmarks.",
        },
    )
    return best_path, history
