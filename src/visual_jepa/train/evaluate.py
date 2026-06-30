from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common.config import get_device_name
from common.metrics import anomaly_metrics, flatten_metrics
from common.paths import ensure_dir
from common.plots import save_heatmap_overlay
from common.reports import markdown_table, write_json, write_markdown_report
from visual_jepa.data.mvtec_ad import prepare_mvtec_from_config
from visual_jepa.data.transforms import denormalize_image
from visual_jepa.models.baselines import pixel_stat_baseline
from visual_jepa.train.pretrain import build_visual_model_from_config


def load_pretrained_visual(cfg: dict[str, Any], device: str):
    ckpt = torch.load(cfg["outputs"]["checkpoint"], map_location=device, weights_only=False)
    model = build_visual_model_from_config(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


@torch.no_grad()
def _latent_prediction_scores(model, x: torch.Tensor, repeats: int = 4) -> np.ndarray:
    vals = []
    for xi in x:
        sample_scores = []
        for _ in range(repeats):
            sample_scores.append(float(model(xi.unsqueeze(0))["pred_loss"].detach().cpu()))
        vals.append(float(np.mean(sample_scores)))
    return np.asarray(vals, dtype=np.float32)


def _collect_train_features(model, dataset, device: str, batch_size: int, max_patch_memory: int = 2048):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    globals_, patches, pred_scores = [], [], []
    for batch in loader:
        x = batch["image"].to(device)
        z = F.normalize(model.encode(x), dim=-1)
        fmap = model.feature_map(x)
        fmap = F.normalize(fmap.permute(0, 2, 3, 1).reshape(-1, fmap.shape[1]), dim=-1)
        globals_.append(z.cpu())
        patches.append(fmap.cpu())
        pred_scores.extend(_latent_prediction_scores(model, x, repeats=2).tolist())
    globals_t = torch.cat(globals_, dim=0)
    patch_t = torch.cat(patches, dim=0)
    if len(patch_t) > max_patch_memory:
        idx = torch.randperm(len(patch_t))[:max_patch_memory]
        patch_t = patch_t[idx]
    center = F.normalize(globals_t.mean(dim=0, keepdim=True), dim=-1)
    return globals_t.to(device), patch_t.to(device), center.to(device), np.asarray(pred_scores, dtype=np.float32)


@torch.no_grad()
def evaluate_visual_jepa(cfg: dict[str, Any], save_overlays: int = 6, include_baseline: bool = True) -> list[dict[str, Any]]:
    device = get_device_name(cfg.get("device", "auto"))
    bundle = prepare_mvtec_from_config(cfg, force=False)
    model = load_pretrained_visual(cfg, device)
    batch_size = int(cfg["training"].get("batch_size", 24))
    train_global, patch_memory, center, train_scores = _collect_train_features(model, bundle.train_dataset, device, batch_size)
    threshold = float(np.quantile(train_scores, 0.95))
    out_root = Path(cfg["outputs"]["root"])
    heatmap_dir = out_root / "heatmaps" / cfg["data"]["category"]
    ensure_dir(heatmap_dir)
    labels, scores, pixel_scores, pixel_labels = [], [], [], []
    rows: list[dict[str, Any]] = []
    start = time.time()
    saved = 0
    for batch in DataLoader(bundle.test_dataset, batch_size=1, shuffle=False):
        x = batch["image"].to(device)
        z = F.normalize(model.encode(x), dim=-1)
        center_score = (1 - (z * center).sum(dim=1)).cpu().numpy()[0]
        pred_error = float(_latent_prediction_scores(model, x, repeats=4)[0])
        fmap = model.feature_map(x)
        _, c, h, w = fmap.shape
        patch = F.normalize(fmap.permute(0, 2, 3, 1).reshape(-1, c), dim=-1)
        dist = 1 - patch @ patch_memory.T
        heat = dist.min(dim=1).values.reshape(1, 1, h, w)
        heat = F.interpolate(heat, size=(cfg["data"]["image_size"], cfg["data"]["image_size"]), mode="bilinear", align_corners=False)
        heat_np = heat.squeeze().cpu().numpy()
        image_score = pred_error
        label = int(batch["label"].item())
        labels.append(label)
        scores.append(image_score)
        pixel_scores.append(heat_np.reshape(-1))
        pixel_labels.append(batch["mask"].numpy().reshape(-1))
        if saved < save_overlays and label == 1:
            img_np = denormalize_image(batch["image"][0]).numpy()
            mask_np = batch["mask"][0, 0].numpy()
            save_heatmap_overlay(
                img_np,
                heat_np,
                mask_np,
                heatmap_dir / f"sample_{saved:03d}_jepa_overlay.png",
                title="Visual-JEPA",
            )
            saved += 1
    metrics = flatten_metrics(anomaly_metrics(np.array(labels), np.array(scores), threshold=threshold, prefix="image_"))
    try:
        metrics.update(flatten_metrics(anomaly_metrics(np.concatenate(pixel_labels), np.concatenate(pixel_scores), prefix="pixel_")))
    except Exception:
        pass
    metrics.update(
        {
            "dataset": "mvtec_ad",
            "category": cfg["data"]["category"],
            "model_name": "visual_jepa_embedding_knn",
            "model_family": "jepa",
            "seed": cfg.get("seed", 42),
            "train_mode": "self_supervised",
            "label_fraction": 0.0,
            "anomaly_method": "latent_prediction_error_plus_encoder_patch_knn",
            "train_time_sec": 0.0,
            "inference_time_ms": (time.time() - start) * 1000 / max(len(bundle.test_dataset), 1),
        }
    )
    rows.append(metrics)
    if include_baseline:
        baseline = pixel_stat_baseline(bundle.train_dataset, bundle.test_dataset, batch_size=batch_size, device=device)
        baseline.update({"dataset": "mvtec_ad", "category": cfg["data"]["category"], "seed": cfg.get("seed", 42)})
        rows.append(baseline)
    ensure_dir(out_root / "benchmark")
    df = pd.DataFrame(rows)
    benchmark_dir = out_root / "benchmark"
    df.to_csv(benchmark_dir / "visual_benchmark_results.csv", index=False)
    ranking = df.sort_values("image_AUROC", ascending=False, na_position="last").copy()
    ranking.insert(0, "rank", range(1, len(ranking) + 1))
    ranking_cols = ["rank", "model_name", "model_family", "image_AUROC", "image_AUPRC", "pixel_AUROC", "pixel_AUPRC"]
    (benchmark_dir / "model_ranking.md").write_text(markdown_table(ranking[ranking_cols].to_dict("records")), encoding="utf-8")
    write_json(
        benchmark_dir / "visual_benchmark_summary.json",
        {
            "dataset": "mvtec_ad",
            "category": cfg["data"]["category"],
            "primary_metric": "image_AUROC",
            "best_model": ranking.iloc[0]["model_name"] if len(ranking) else None,
            "best_image_AUROC": float(ranking.iloc[0]["image_AUROC"]) if len(ranking) else None,
            "sota_claim": False,
            "pending_strong_baselines": ["PatchCore", "PaDiM"],
        },
    )
    write_markdown_report(
        out_root / "reports" / "visual_eval_report.md",
        "Visual-JEPA Evaluation",
        {
            "Metrics": markdown_table(rows),
            "Ranking": markdown_table(ranking[ranking_cols].to_dict("records")),
            "Heatmaps": f"Saved overlays under `{heatmap_dir}`.",
            "No SOTA Claim": "PatchCore/PaDiM and published-number comparisons are pending.",
        },
    )
    write_markdown_report(
        benchmark_dir / "reports" / "visual_benchmark_report.md",
        "Visual Benchmark Report",
        {
            "Results": markdown_table(rows),
            "Ranking": markdown_table(ranking[ranking_cols].to_dict("records")),
            "Pending Strong Baselines": "PatchCore and PaDiM are pending because anomalib is not installed.",
        },
    )
    return rows
