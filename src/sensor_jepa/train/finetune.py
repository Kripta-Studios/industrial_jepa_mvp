from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from common.config import get_device_name
from common.metrics import classification_metrics, flatten_metrics
from common.paths import ensure_dir
from common.reports import markdown_table, write_markdown_report
from sensor_jepa.data.cnc_milling import prepare_from_config
from sensor_jepa.data.windowing import stratified_label_fraction_indices
from sensor_jepa.models.heads import ClassificationHead
from sensor_jepa.train.probe import load_pretrained_sensor


class SensorClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, embedding_dim: int = 128, num_classes: int = 3):
        super().__init__()
        self.encoder = encoder
        self.head = ClassificationHead(embedding_dim, num_classes, hidden_dim=128)

    def forward(self, x):
        return self.head(self.encoder(x))


def finetune_sensor_jepa(
    cfg: dict[str, Any],
    mode: str = "full",
    label_fraction: float | None = None,
) -> dict[str, Any]:
    device = get_device_name(cfg.get("device", "auto"))
    bundle = prepare_from_config(cfg, force=False)
    base = load_pretrained_sensor(cfg, device)
    embedding_dim = int(cfg["model"].get("embedding_dim", 128))
    model = SensorClassifier(base.encoder, embedding_dim=embedding_dim, num_classes=len(bundle.class_names)).to(device)
    frozen = mode == "frozen"
    if frozen:
        for p in model.encoder.parameters():
            p.requires_grad = False
    elif mode == "semi":
        for name, p in model.encoder.named_parameters():
            p.requires_grad = "proj" in name
    if label_fraction is None:
        label_fraction = float(cfg["training"].get("label_fraction", 1.0))
    idx = stratified_label_fraction_indices(bundle.y_train, label_fraction, seed=int(cfg.get("seed", 42)))
    x_train, y_train = bundle.x_train[idx], bundle.y_train[idx]
    ds = TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    loader = DataLoader(ds, batch_size=int(cfg["training"].get("batch_size", 64)), shuffle=True)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(cfg["training"].get("learning_rate", 1e-3)),
        weight_decay=float(cfg["training"].get("weight_decay", 1e-4)),
    )
    loss_fn = nn.CrossEntropyLoss()
    start = time.time()
    for _ in range(int(cfg["training"].get("finetune_epochs", 10))):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(bundle.x_test, dtype=torch.float32, device=device)).cpu().numpy()
    pred = logits.argmax(axis=1)
    metrics = flatten_metrics(classification_metrics(bundle.y_test, pred, logits))
    metrics.update(
        {
            "dataset": "cnc_milling",
            "task": "wear_classification",
            "model_name": f"sensor_jepa_{mode}_finetune",
            "model_family": "jepa",
            "seed": cfg.get("seed", 42),
            "label_fraction": label_fraction,
            "encoder_mode": mode,
            "probe_type": "finetune",
            "frozen_encoder": frozen,
            "train_time_sec": time.time() - start,
        }
    )
    out_root = Path(cfg["outputs"]["root"])
    ensure_dir(out_root / "finetune")
    pd.DataFrame([metrics]).to_csv(out_root / "finetune" / f"sensor_finetune_{mode}_{label_fraction}.csv", index=False)
    write_markdown_report(
        out_root / "reports" / f"sensor_finetune_{mode}_{label_fraction}.md",
        "Sensor-JEPA Fine-tune",
        {"Metrics": markdown_table([metrics])},
    )
    return metrics

