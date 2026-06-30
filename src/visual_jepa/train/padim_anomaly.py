from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from common.config import get_device_name
from common.paths import ensure_dir
from common.plots import save_heatmap_overlay
from common.reports import markdown_table, write_json, write_markdown_report
from visual_jepa.data.industrial import prepare_dense_visual_data
from visual_jepa.data.transforms import load_rgb
from visual_jepa.eval.dense_feature_eval import binary_operating_metrics, pixel_overlap_metrics, quantile_threshold
from visual_jepa.models.padim_lite import PadimLite
from visual_jepa.train.extract_dense_features import _extract_split, build_feature_backbone
from visual_jepa.train.memory_anomaly import _subset


def _heatmaps(scores: torch.Tensor, grid_shape: tuple[int, int], image_size: int) -> torch.Tensor:
    return F.interpolate(scores.reshape(scores.shape[0], 1, grid_shape[0], grid_shape[1]), size=(image_size, image_size), mode="bilinear", align_corners=False)


def evaluate_visual_padim(cfg: dict[str, Any], backbone: str = "dense_visual_jepa") -> dict[str, Path]:
    device = get_device_name(cfg.get("device", "auto"))
    bundle = prepare_dense_visual_data(cfg)
    batch_size = int(cfg.get("train", {}).get("batch_size", 32))
    eval_cfg = cfg.get("padim", {})
    out_root = ensure_dir(eval_cfg.get("output_dir", "outputs/visual_jepa/padim_benchmark"))
    model, info = build_feature_backbone(cfg, backbone, device)
    if model is None:
        path = out_root / "results.csv"
        pd.DataFrame([{"model_name": f"{backbone}_padim", "status": "unavailable", "notes": info.get("message", "")}]).to_csv(path, index=False)
        return {"results": path, "report": out_root / "report.md"}

    start = time.time()
    train_data = _extract_split(model, bundle.train_dataset, "train", device, batch_size, backbone)
    val_data = _extract_split(model, bundle.val_dataset, "val", device, batch_size, backbone)
    test_data = _extract_split(model, bundle.test_dataset, "test", device, batch_size, backbone)
    test_df = pd.DataFrame(test_data["rows"])
    train_df = pd.DataFrame(train_data["rows"])
    val_df = pd.DataFrame(val_data["rows"])
    image_size = int(cfg.get("dataset", {}).get("image_size", 224))
    rows = []
    heatmap_dir = ensure_dir(out_root / "heatmaps" / backbone)
    for (dataset, category), group in test_df.groupby(["dataset", "category"], dropna=False):
        train_idx = train_df[(train_df["dataset"].eq(dataset)) & (train_df["category"].eq(category))].index.to_numpy()
        val_idx = val_df[(val_df["dataset"].eq(dataset)) & (val_df["category"].eq(category))].index.to_numpy()
        test_idx = group.index.to_numpy()
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        tr = _subset(train_data, train_idx)
        va = _subset(val_data, val_idx) if len(val_idx) else tr
        te = _subset(test_data, test_idx)
        scorer = PadimLite(
            n_features=eval_cfg.get("n_features", 100),
            eps=float(eval_cfg.get("eps", 1e-3)),
            seed=int(cfg.get("seed", 42)),
            top_k=int(eval_cfg.get("top_k", 5)),
        ).fit(tr["embeddings"].to(device))
        val_scores = scorer.score(va["embeddings"].to(device))
        test_scores = scorer.score(te["embeddings"].to(device))
        image_threshold = quantile_threshold(val_scores.image_scores.cpu().numpy(), q=float(eval_cfg.get("image_threshold_quantile", 0.99)))
        pixel_threshold = quantile_threshold(val_scores.patch_scores.cpu().numpy().reshape(-1), q=float(eval_cfg.get("pixel_threshold_quantile", 0.995)))
        heat = _heatmaps(test_scores.patch_scores.cpu(), te["grid_shape"], image_size)
        row = binary_operating_metrics(te["labels"].numpy(), test_scores.image_scores.cpu().numpy(), image_threshold, prefix="image_")
        try:
            row.update(pixel_overlap_metrics(te["masks"].numpy(), heat.numpy(), pixel_threshold, prefix="pixel_"))
        except Exception:
            pass
        row.update(
            {
                "dataset": dataset,
                "category": category,
                "model_name": f"{backbone}_padim",
                "model_family": "padim_lite",
                "anomaly_method": "position_gaussian_mahalanobis",
                "threshold_source": "normal_validation_quantile",
                "feature_dim": int(min(tr["embeddings"].shape[-1], eval_cfg.get("n_features", tr["embeddings"].shape[-1]) or tr["embeddings"].shape[-1])),
                "grid_shape": str(te["grid_shape"]),
                "train_time_sec": time.time() - start,
                "status": "ok",
            }
        )
        rows.append(row)
        saved = 0
        for i, meta in enumerate(te["rows"]):
            if saved >= int(eval_cfg.get("save_heatmaps", 4)):
                break
            if int(meta["label"]) != 1:
                continue
            img = load_rgb(meta["image_path"], image_size).numpy()
            save_heatmap_overlay(img, heat[i, 0].numpy(), te["masks"][i].numpy(), heatmap_dir / str(category) / f"sample_{saved:03d}.png", title="PaDiM-lite")
            saved += 1

    results = pd.DataFrame(rows)
    results_path = out_root / "results.csv"
    results.to_csv(results_path, index=False)
    write_json(out_root / "backbone_info.json", info)
    write_markdown_report(
        out_root / "report.md",
        "Visual PaDiM-lite Benchmark",
        {
            "Backbone": f"`{backbone}`\n\n`{info}`",
            "Results": markdown_table(results.to_dict("records")),
            "Conclusion Rules": "This is a lightweight PaDiM-style implementation, not an official PaDiM reproduction.",
        },
    )
    return {"results": results_path, "report": out_root / "report.md"}
