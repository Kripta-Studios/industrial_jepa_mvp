from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

from common.config import get_device_name
from common.paths import ensure_dir
from common.plots import plot_history
from common.reports import write_markdown_report
from common.seed import seed_everything
from visual_jepa.data.mvtec_ad import prepare_mvtec_from_config
from visual_jepa.models.visual_jepa import VisualJEPA


def build_visual_model_from_config(cfg: dict[str, Any]) -> VisualJEPA:
    m = cfg["model"]
    return VisualJEPA(
        encoder=m.get("encoder", "small_conv"),
        embedding_dim=int(m.get("embedding_dim", 128)),
        hidden_dim=int(m.get("hidden_dim", 128)),
        predictor_hidden_dim=int(m.get("predictor_hidden_dim", 256)),
        patch_size=int(m.get("patch_size", 16)),
        mask_ratio=float(m.get("mask_ratio", 0.5)),
        sigreg_weight=float(m.get("sigreg_weight", 0.05)),
    )


def pretrain_visual_jepa(cfg: dict[str, Any], force_data: bool = False) -> tuple[Path, list[dict[str, float]]]:
    seed_everything(int(cfg.get("seed", 42)))
    device = get_device_name(cfg.get("device", "auto"))
    bundle = prepare_mvtec_from_config(cfg, force=force_data)
    model = build_visual_model_from_config(cfg).to(device)
    train_cfg = cfg["training"]
    loader = DataLoader(
        bundle.train_dataset,
        batch_size=int(train_cfg.get("batch_size", 24)),
        shuffle=True,
        num_workers=int(cfg["data"].get("num_workers", 0)),
    )
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    history: list[dict[str, float]] = []
    start = time.time()
    for epoch in range(1, int(train_cfg.get("pretrain_epochs", 4)) + 1):
        model.train()
        agg = {"loss": 0.0, "pred_loss": 0.0, "sigreg": 0.0, "embedding_std": 0.0}
        n = 0
        for batch in loader:
            xb = batch["image"].to(device)
            opt.zero_grad(set_to_none=True)
            out = model(xb)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            bs = len(xb)
            n += bs
            for k in agg:
                agg[k] += float(out[k].detach().cpu()) * bs
        history.append({"epoch": epoch, **{k: v / max(n, 1) for k, v in agg.items()}})
    ckpt_path = Path(cfg["outputs"]["checkpoint"])
    ensure_dir(ckpt_path.parent)
    torch.save(
        {
            "model_state": model.state_dict(),
            "cfg": cfg,
            "history": history,
        },
        ckpt_path,
    )
    out_root = Path(cfg["outputs"]["root"])
    ensure_dir(out_root)
    pd.DataFrame(history).to_csv(out_root / "pretrain_history.csv", index=False)
    plot_history(history, out_root / "plots" / "visual_pretrain_loss.png", y_key="loss")
    write_markdown_report(
        out_root / "reports" / "visual_pretrain_report.md",
        "Visual-JEPA Pretraining",
        {
            "Setup": f"Dataset: MVTec AD `{cfg['data']['category']}`\n\nDevice: `{device}`\n\nTrain images: `{len(bundle.train_dataset)}`",
            "Result": f"Final loss: `{history[-1]['loss']:.6f}`\n\nElapsed seconds: `{time.time() - start:.2f}`",
            "No SOTA Claim": "This is an MVP run and is not validated as SOTA.",
        },
    )
    return ckpt_path, history

