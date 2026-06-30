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
from visual_jepa.models.patchcore_lite import PatchCoreLite
from visual_jepa.train.extract_dense_features import _extract_split, build_feature_backbone


def _subset(data: dict[str, Any], idx: np.ndarray) -> dict[str, Any]:
    t_idx = torch.tensor(idx, dtype=torch.long)
    return {
        "embeddings": data["embeddings"][t_idx],
        "labels": data["labels"][t_idx],
        "masks": data["masks"][t_idx],
        "rows": [data["rows"][int(i)] for i in idx],
        "grid_shape": data["grid_shape"],
    }


def _heatmaps_from_patch_scores(scores: torch.Tensor, grid_shape: tuple[int, int], image_size: int) -> torch.Tensor:
    b = scores.shape[0]
    heat = scores.reshape(b, 1, grid_shape[0], grid_shape[1])
    return F.interpolate(heat, size=(image_size, image_size), mode="bilinear", align_corners=False)


def _save_heatmaps(
    rows: list[dict[str, Any]],
    heatmaps: torch.Tensor,
    masks: torch.Tensor,
    out_dir: Path,
    max_items: int = 6,
) -> None:
    saved = 0
    for i, row in enumerate(rows):
        if saved >= max_items:
            break
        if int(row["label"]) != 1:
            continue
        img = load_rgb(row["image_path"], heatmaps.shape[-1]).numpy()
        save_heatmap_overlay(
            img,
            heatmaps[i, 0].cpu().numpy(),
            masks[i].cpu().numpy(),
            out_dir / f"{row['dataset']}_{row['category']}_{saved:03d}.png",
            title="PatchCore-lite",
        )
        saved += 1


def evaluate_visual_memory_anomaly(cfg: dict[str, Any], backbone: str = "dense_visual_jepa") -> dict[str, Path]:
    device = get_device_name(cfg.get("device", "auto"))
    bundle = prepare_dense_visual_data(cfg)
    batch_size = int(cfg.get("train", {}).get("batch_size", 32))
    eval_cfg = cfg.get("memory", {})
    out_root = ensure_dir(eval_cfg.get("output_dir", "outputs/visual_jepa/memory_benchmark"))
    model, info = build_feature_backbone(cfg, backbone, device)
    if model is None:
        results = pd.DataFrame(
            [
                {
                    "model_name": f"{backbone}_knn",
                    "model_family": "unavailable",
                    "status": "unavailable",
                    "notes": info.get("message", "backbone unavailable"),
                }
            ]
        )
        path = out_root / "results.csv"
        results.to_csv(path, index=False)
        write_json(out_root / "backbone_info.json", info)
        return {"results": path, "report": out_root / "report.md"}

    start_extract = time.time()
    train_data = _extract_split(model, bundle.train_dataset, "train", device, batch_size, backbone)
    val_data = _extract_split(model, bundle.val_dataset, "val", device, batch_size, backbone)
    test_data = _extract_split(model, bundle.test_dataset, "test", device, batch_size, backbone)
    rows = []
    per_category_rows = []
    heatmap_dir = ensure_dir(out_root / "heatmaps" / backbone)
    test_df = pd.DataFrame(test_data["rows"])
    train_df = pd.DataFrame(train_data["rows"])
    val_df = pd.DataFrame(val_data["rows"])
    image_size = int(cfg.get("dataset", {}).get("image_size", 224))
    for (dataset, category), group in test_df.groupby(["dataset", "category"], dropna=False):
        train_idx = train_df[(train_df["dataset"].eq(dataset)) & (train_df["category"].eq(category))].index.to_numpy()
        val_idx = val_df[(val_df["dataset"].eq(dataset)) & (val_df["category"].eq(category))].index.to_numpy()
        test_idx = group.index.to_numpy()
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        tr = _subset(train_data, train_idx)
        va = _subset(val_data, val_idx) if len(val_idx) else tr
        te = _subset(test_data, test_idx)
        scorer = PatchCoreLite(
            coreset_ratio=float(eval_cfg.get("coreset_ratio", 0.1)),
            coreset_method=eval_cfg.get("coreset_method", "random"),
            top_k=int(eval_cfg.get("top_k", 5)),
            seed=int(cfg.get("seed", 42)),
            max_memory_patches=eval_cfg.get("max_memory_patches", 20000),
        ).fit(tr["embeddings"].to(device))
        val_scores = scorer.score(va["embeddings"].to(device))
        test_scores = scorer.score(te["embeddings"].to(device))
        image_threshold = quantile_threshold(val_scores.image_scores.cpu().numpy(), q=float(eval_cfg.get("image_threshold_quantile", 0.99)))
        pixel_threshold = quantile_threshold(val_scores.patch_scores.cpu().numpy().reshape(-1), q=float(eval_cfg.get("pixel_threshold_quantile", 0.995)))
        heat = _heatmaps_from_patch_scores(test_scores.patch_scores.cpu(), te["grid_shape"], image_size)
        row = binary_operating_metrics(te["labels"].numpy(), test_scores.image_scores.cpu().numpy(), image_threshold, prefix="image_")
        try:
            row.update(pixel_overlap_metrics(te["masks"].numpy(), heat.numpy(), pixel_threshold, prefix="pixel_"))
        except Exception:
            pass
        row.update(
            {
                "dataset": dataset,
                "category": category,
                "model_name": f"{backbone}_knn",
                "model_family": "patchcore_lite",
                "anomaly_method": "patch_memory_knn_topk",
                "threshold_source": "normal_validation_quantile",
                "memory_bank_size": int(len(scorer.memory)) if scorer.memory is not None else 0,
                "feature_dim": int(tr["embeddings"].shape[-1]),
                "grid_shape": str(te["grid_shape"]),
                "train_time_sec": time.time() - start_extract,
                "inference_time_ms": 0.0,
                "status": "ok",
            }
        )
        rows.append(row)
        per_category_rows.append(row.copy())
        _save_heatmaps(te["rows"], heat, te["masks"], heatmap_dir / str(category), max_items=int(eval_cfg.get("save_heatmaps", 4)))

    results = pd.DataFrame(rows)
    results_path = out_root / "results.csv"
    per_category_path = out_root / "per_category_metrics.csv"
    results.to_csv(results_path, index=False)
    pd.DataFrame(per_category_rows).to_csv(per_category_path, index=False)
    write_json(out_root / "backbone_info.json", info)
    write_markdown_report(
        out_root / "report.md",
        "Visual Memory Anomaly Benchmark",
        {
            "Backbone": f"`{backbone}`\n\n`{info}`",
            "Results": markdown_table(results.to_dict("records")),
            "Conclusion Rules": "This is PatchCore-lite/kNN, not an official PatchCore implementation.",
        },
    )
    return {"results": results_path, "per_category": per_category_path, "report": out_root / "report.md"}
