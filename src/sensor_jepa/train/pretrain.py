from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from common.config import get_device_name
from common.paths import ensure_dir
from common.plots import plot_history
from common.reports import write_markdown_report
from common.seed import seed_everything
from sensor_jepa.data.cnc_milling import prepare_from_config
from sensor_jepa.models.sensor_jepa import SensorJEPA


def build_model_from_config(cfg: dict[str, Any], input_channels: int) -> SensorJEPA:
    m = cfg["model"]
    return SensorJEPA(
        input_channels=input_channels,
        encoder=m.get("encoder", "conv1d"),
        embedding_dim=int(m.get("embedding_dim", 128)),
        hidden_dim=int(m.get("hidden_dim", 128)),
        predictor_hidden_dim=int(m.get("predictor_hidden_dim", 256)),
        temporal_mask_ratio=float(m.get("temporal_mask_ratio", 0.4)),
        channel_mask_ratio=float(m.get("channel_mask_ratio", 0.15)),
        sigreg_weight=float(m.get("sigreg_weight", 0.05)),
    )


def pretrain_sensor_jepa(cfg: dict[str, Any], force_data: bool = False) -> tuple[Path, list[dict[str, float]]]:
    seed_everything(int(cfg.get("seed", 42)))
    device = get_device_name(cfg.get("device", "auto"))
    bundle = prepare_from_config(cfg, force=force_data)
    model = build_model_from_config(cfg, bundle.input_channels).to(device)
    train_cfg = cfg["training"]
    ds = TensorDataset(torch.tensor(bundle.x_train, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=int(train_cfg.get("batch_size", 64)), shuffle=True, drop_last=False)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    history: list[dict[str, float]] = []
    start = time.time()
    for epoch in range(1, int(train_cfg.get("pretrain_epochs", 8)) + 1):
        model.train()
        agg = {"loss": 0.0, "pred_loss": 0.0, "sigreg": 0.0, "embedding_std": 0.0}
        n = 0
        for (xb,) in loader:
            xb = xb.to(device)
            opt.zero_grad(set_to_none=True)
            out = model(xb)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            bs = len(xb)
            n += bs
            for k in agg:
                agg[k] += float(out[k].detach().cpu()) * bs
        row = {"epoch": epoch, **{k: v / max(n, 1) for k, v in agg.items()}}
        history.append(row)

    ckpt_path = Path(cfg["outputs"]["checkpoint"])
    ensure_dir(ckpt_path.parent)
    torch.save(
        {
            "model_state": model.state_dict(),
            "cfg": cfg,
            "input_channels": bundle.input_channels,
            "class_names": bundle.class_names,
            "feature_names": bundle.feature_names,
            "history": history,
        },
        ckpt_path,
    )
    out_root = Path(cfg["outputs"]["root"])
    ensure_dir(out_root)
    pd.DataFrame(history).to_csv(out_root / "pretrain_history.csv", index=False)
    plot_history(history, out_root / "plots" / "sensor_pretrain_loss.png", y_key="loss")
    elapsed = time.time() - start
    write_markdown_report(
        out_root / "reports" / "sensor_pretrain_report.md",
        "Sensor-JEPA Pretraining",
        {
            "Setup": f"Dataset: CNC milling feature windows\n\nDevice: `{device}`\n\nTrain windows: `{len(bundle.x_train)}`",
            "Result": f"Final loss: `{history[-1]['loss']:.6f}`\n\nElapsed seconds: `{elapsed:.2f}`",
            "No SOTA Claim": "This run is an MVP pretraining run, not a SOTA validation.",
        },
    )
    return ckpt_path, history

