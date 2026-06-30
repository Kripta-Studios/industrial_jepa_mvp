from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from common.config import get_device_name
from common.paths import ensure_dir
from common.reports import write_markdown_report
from common.seed import seed_everything
from sensor_jepa.data.cnc_world_model import prepare_transition_from_config
from sensor_jepa.eval.dense_sensor_surprise import build_dense_model_from_config


def _run_epoch(model, loader, device: str, optimizer=None) -> dict[str, float]:
    train = optimizer is not None
    model.train(train)
    totals: dict[str, float] = {}
    n = 0
    for xb, xnb in loader:
        xb = xb.to(device)
        xnb = xnb.to(device)
        with torch.set_grad_enabled(train):
            out = model(xb, x_future=xnb)
            loss = out["loss"]
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                model.update_target_encoder()
        batch = len(xb)
        n += batch
        for key in [
            "loss",
            "target_loss",
            "visible_loss",
            "future_loss",
            "variance_loss",
            "sigreg_loss",
            "embedding_std",
            "collapse_score",
            "effective_rank_ratio",
        ]:
            value = out[key]
            totals[key] = totals.get(key, 0.0) + float(value.detach().cpu()) * batch
    return {key: value / max(n, 1) for key, value in totals.items()}


def pretrain_dense_sensor_jepa(cfg: dict[str, Any]) -> dict[str, Path]:
    seed_everything(int(cfg.get("seed", 42)))
    device = get_device_name(cfg.get("device", "auto"))
    bundle = prepare_transition_from_config(cfg)
    model = build_dense_model_from_config(cfg, bundle.input_channels).to(device)
    train_cfg = cfg.get("train", {})
    out_dir = ensure_dir(cfg.get("eval", {}).get("output_dir", cfg.get("outputs", {}).get("root", "outputs/sensor_jepa/dense_sensor_jepa_cnc")))
    ckpt_dir = ensure_dir(out_dir / "checkpoints")
    batch_size = int(train_cfg.get("batch_size", 64))
    train_loader = DataLoader(
        TensorDataset(torch.tensor(bundle.x_train, dtype=torch.float32), torch.tensor(bundle.x_next_train, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(bundle.x_val, dtype=torch.float32), torch.tensor(bundle.x_next_val, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=False,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", train_cfg.get("learning_rate", 3e-4))),
        weight_decay=float(train_cfg.get("weight_decay", 1e-2)),
    )
    epochs = int(train_cfg.get("epochs", 5))
    masking_cfg = cfg.get("masking", {})
    schedule_cfg = masking_cfg.get("schedule", {})
    schedule_enabled = bool(schedule_cfg.get("enabled", False))
    final_mask_ratio = float(masking_cfg.get("temporal_mask_ratio", model.temporal_mask_ratio))
    initial_mask_ratio = float(schedule_cfg.get("initial_mask_ratio", min(0.15, final_mask_ratio)))
    final_max_span = int(masking_cfg.get("max_span", model.max_span))
    initial_max_span = int(schedule_cfg.get("initial_max_span", max(model.min_span, min(model.max_span, 1))))
    history = []
    best_val = float("inf")
    best_path = ckpt_dir / "best.pt"
    latest_path = ckpt_dir / "latest.pt"
    for epoch in range(1, epochs + 1):
        if schedule_enabled:
            progress = 1.0 if epochs <= 1 else (epoch - 1) / max(epochs - 1, 1)
            model.temporal_mask_ratio = initial_mask_ratio + (final_mask_ratio - initial_mask_ratio) * progress
            model.max_span = int(round(initial_max_span + (final_max_span - initial_max_span) * progress))
        train_metrics = _run_epoch(model, train_loader, device, optimizer=optimizer)
        with torch.no_grad():
            val_metrics = _run_epoch(model, val_loader, device, optimizer=None)
        row = {"epoch": epoch}
        row.update({f"train_{k}": v for k, v in train_metrics.items()})
        row.update({f"val_{k}": v for k, v in val_metrics.items()})
        history.append(row)
        torch.save({"model_state": model.state_dict(), "config": cfg, "epoch": epoch}, latest_path)
        if val_metrics.get("loss", float("inf")) < best_val:
            best_val = val_metrics["loss"]
            torch.save({"model_state": model.state_dict(), "config": cfg, "epoch": epoch}, best_path)
    metrics_path = out_dir / "pretrain_metrics.csv"
    pd.DataFrame(history).to_csv(metrics_path, index=False)
    report_path = out_dir / "report.md"
    write_markdown_report(
        report_path,
        "DenseSensorJEPA Pretrain Report",
        {
            "Status": "Pretraining completed. This is representation training only; MVP evidence depends on incremental delta over metadata/cycle.",
            "Best Validation Loss": f"{best_val:.6f}",
            "Checkpoints": f"`{latest_path}`\n\n`{best_path}`",
            "Metrics": f"`{metrics_path}`",
        },
    )
    return {"latest_checkpoint": latest_path, "best_checkpoint": best_path, "metrics": metrics_path, "report": report_path}
