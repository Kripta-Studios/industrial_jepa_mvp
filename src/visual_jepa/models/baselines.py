from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common.metrics import anomaly_metrics, flatten_metrics


@torch.no_grad()
def pixel_stat_baseline(train_dataset, test_dataset, batch_size: int = 16, device: str = "cpu") -> dict[str, Any]:
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    imgs = []
    for batch in train_loader:
        imgs.append(batch["image"])
    train = torch.cat(imgs, dim=0).to(device)
    mean = train.mean(dim=0, keepdim=True)
    std = train.std(dim=0, keepdim=True).clamp_min(0.1)
    train_scores = ((train - mean).abs() / std).mean(dim=(1, 2, 3)).cpu().numpy()
    threshold = float(np.quantile(train_scores, 0.95))

    scores, labels, pixel_scores, pixel_labels = [], [], [], []
    for batch in DataLoader(test_dataset, batch_size=batch_size, shuffle=False):
        x = batch["image"].to(device)
        heat = ((x - mean).abs() / std).mean(dim=1, keepdim=True)
        score = heat.mean(dim=(1, 2, 3)).cpu().numpy()
        scores.extend(score.tolist())
        labels.extend(batch["label"].numpy().tolist())
        pixel_scores.append(heat.cpu().numpy().reshape(-1))
        pixel_labels.append(batch["mask"].numpy().reshape(-1))
    row = flatten_metrics(anomaly_metrics(np.array(labels), np.array(scores), threshold=threshold, prefix="image_"))
    try:
        row.update(flatten_metrics(anomaly_metrics(np.concatenate(pixel_labels), np.concatenate(pixel_scores), prefix="pixel_")))
    except Exception:
        pass
    row.update({"model_name": "pixel_stat_baseline", "model_family": "classic", "anomaly_method": "pixel_zscore"})
    return row

