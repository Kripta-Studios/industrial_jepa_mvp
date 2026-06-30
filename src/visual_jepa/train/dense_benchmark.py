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
from common.paths import ensure_dir
from common.reports import markdown_table, write_markdown_report
from visual_jepa.data.industrial import IndustrialVisualDataset, prepare_dense_visual_data
from visual_jepa.eval.dense_feature_eval import binary_operating_metrics, pixel_overlap_metrics, quantile_threshold
from visual_jepa.models.baselines import pixel_stat_baseline
from visual_jepa.models.dense_visual_jepa import build_dense_visual_jepa_from_config
from visual_jepa.train.extract_dense_features import _dense_checkpoint_path
from visual_jepa.train.memory_anomaly import evaluate_visual_memory_anomaly
from visual_jepa.train.padim_anomaly import evaluate_visual_padim


def _category_dataset(df: pd.DataFrame, dataset: str, category: str, image_size: int) -> IndustrialVisualDataset:
    sub = df[(df["dataset"].eq(dataset)) & (df["category"].eq(category))].copy()
    return IndustrialVisualDataset(sub, image_size=image_size)


@torch.no_grad()
def _dense_latent_rows(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    device = get_device_name(cfg.get("device", "auto"))
    bundle = prepare_dense_visual_data(cfg)
    model = build_dense_visual_jepa_from_config(cfg).to(device)
    ckpt = _dense_checkpoint_path(cfg)
    pretrained = False
    if ckpt is not None:
        state = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state"], strict=False)
        pretrained = True
    model.eval()
    image_size = int(cfg.get("dataset", {}).get("image_size", 224))
    batch_size = int(cfg.get("train", {}).get("batch_size", 32))
    repeats = int(cfg.get("benchmark", {}).get("latent_repeats", 2))
    rows = []
    test_df = bundle.test_dataset.df
    val_df = bundle.val_dataset.df

    def score_dataset(dataset: IndustrialVisualDataset):
        image_scores, labels, heats, masks = [], [], [], []
        for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            x = batch["image"].to(device)
            heat_acc = None
            for _ in range(repeats):
                patch_heat, grid = model.latent_error_map(x)
                heat = F.interpolate(patch_heat, size=(image_size, image_size), mode="bilinear", align_corners=False)
                heat_acc = heat if heat_acc is None else heat_acc + heat
            heat = (heat_acc / max(repeats, 1)).cpu()
            heats.append(heat)
            image_scores.extend(heat.flatten(1).topk(max(1, min(32, heat.shape[-1] * heat.shape[-2])), dim=1).values.mean(dim=1).numpy().tolist())
            labels.extend(batch["label"].numpy().tolist())
            masks.append(batch["mask"].squeeze(1).cpu())
        return np.asarray(image_scores), np.asarray(labels), torch.cat(heats, dim=0).numpy(), torch.cat(masks, dim=0).numpy()

    for dataset, category in test_df[["dataset", "category"]].drop_duplicates().itertuples(index=False):
        val_ds = _category_dataset(val_df, dataset, category, image_size)
        test_ds = _category_dataset(test_df, dataset, category, image_size)
        if len(test_ds) == 0:
            continue
        val_scores, _, val_heat, _ = score_dataset(val_ds if len(val_ds) else test_ds)
        test_scores, labels, test_heat, masks = score_dataset(test_ds)
        image_threshold = quantile_threshold(val_scores, q=float(cfg.get("benchmark", {}).get("image_threshold_quantile", 0.99)))
        pixel_threshold = quantile_threshold(val_heat.reshape(-1), q=float(cfg.get("benchmark", {}).get("pixel_threshold_quantile", 0.995)))
        row = binary_operating_metrics(labels, test_scores, image_threshold, prefix="image_")
        try:
            row.update(pixel_overlap_metrics(masks, test_heat[:, 0], pixel_threshold, prefix="pixel_"))
        except Exception:
            pass
        row.update(
            {
                "dataset": dataset,
                "category": category,
                "model_name": "dense_visual_jepa_latent_error",
                "model_family": "dense_jepa",
                "anomaly_method": "latent_prediction_error",
                "checkpoint": str(ckpt) if ckpt else "",
                "pretrained": pretrained,
                "status": "ok",
            }
        )
        rows.append(row)
    return rows


def _pixel_rows(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    device = get_device_name(cfg.get("device", "auto"))
    bundle = prepare_dense_visual_data(cfg)
    image_size = int(cfg.get("dataset", {}).get("image_size", 224))
    batch_size = int(cfg.get("train", {}).get("batch_size", 32))
    rows = []
    for dataset, category in bundle.test_dataset.df[["dataset", "category"]].drop_duplicates().itertuples(index=False):
        train_ds = _category_dataset(bundle.train_dataset.df, dataset, category, image_size)
        test_ds = _category_dataset(bundle.test_dataset.df, dataset, category, image_size)
        if len(train_ds) == 0 or len(test_ds) == 0:
            continue
        row = pixel_stat_baseline(train_ds, test_ds, batch_size=batch_size, device=device)
        row.update({"dataset": dataset, "category": category, "status": "ok"})
        rows.append(row)
    return rows


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if path.exists():
        return pd.read_csv(path).to_dict("records")
    return []


def run_dense_visual_benchmark(cfg: dict[str, Any]) -> dict[str, Path]:
    out_root = ensure_dir(cfg.get("benchmark", {}).get("output_dir", "outputs/visual_jepa/dense_benchmark"))
    all_rows: list[dict[str, Any]] = []
    start = time.time()
    all_rows.extend(_pixel_rows(cfg))
    try:
        all_rows.extend(_dense_latent_rows(cfg))
    except Exception as exc:
        all_rows.append({"model_name": "dense_visual_jepa_latent_error", "status": "failed", "notes": f"{type(exc).__name__}: {exc}"})

    backbones = cfg.get("benchmark", {}).get("memory_backbones", ["dense_visual_jepa", "resnet18"])
    for backbone in backbones:
        mem_cfg = dict(cfg)
        mem_cfg["memory"] = dict(cfg.get("memory", {}))
        mem_cfg["memory"]["output_dir"] = str(out_root / "memory" / backbone)
        try:
            paths = evaluate_visual_memory_anomaly(mem_cfg, backbone=backbone)
            all_rows.extend(_read_rows(Path(paths["results"])))
        except Exception as exc:
            all_rows.append({"model_name": f"{backbone}_knn", "status": "failed", "notes": f"{type(exc).__name__}: {exc}"})

    padim_backbones = cfg.get("benchmark", {}).get("padim_backbones", ["dense_visual_jepa", "resnet18"])
    for backbone in padim_backbones:
        padim_cfg = dict(cfg)
        padim_cfg["padim"] = dict(cfg.get("padim", {}))
        padim_cfg["padim"]["output_dir"] = str(out_root / "padim" / backbone)
        try:
            paths = evaluate_visual_padim(padim_cfg, backbone=backbone)
            all_rows.extend(_read_rows(Path(paths["results"])))
        except Exception as exc:
            all_rows.append({"model_name": f"{backbone}_padim", "status": "failed", "notes": f"{type(exc).__name__}: {exc}"})

    results = pd.DataFrame(all_rows)
    results_path = out_root / "results.csv"
    results.to_csv(results_path, index=False)
    metric_cols = [c for c in ["image_AUROC", "image_AUPRC", "pixel_AUROC", "pixel_AUPRC", "pixel_IoU", "pixel_Dice"] if c in results.columns]
    if metric_cols and "model_name" in results:
        mean_std = results.groupby("model_name", dropna=False)[metric_cols].agg(["mean", "std"]).reset_index()
    else:
        mean_std = pd.DataFrame()
    mean_std_path = out_root / "results_mean_std.csv"
    mean_std.to_csv(mean_std_path, index=False)
    per_category_path = out_root / "per_category_metrics.csv"
    results.to_csv(per_category_path, index=False)
    ranking = results.sort_values("image_AUROC", ascending=False, na_position="last") if "image_AUROC" in results else results
    ranking_path = out_root / "model_ranking.md"
    ranking_path.write_text(markdown_table(ranking.head(50).to_dict("records")), encoding="utf-8")

    def best(name: str) -> str:
        sub = results[results["model_name"].astype(str).str.contains(name, regex=False)] if "model_name" in results else pd.DataFrame()
        if sub.empty or "image_AUROC" not in sub:
            return "not run"
        row = sub.sort_values("image_AUROC", ascending=False, na_position="last").iloc[0]
        return f"{row.get('model_name')} image_AUROC={row.get('image_AUROC')}, image_AUPRC={row.get('image_AUPRC')}"

    write_markdown_report(
        out_root / "report.md",
        "Dense Visual JEPA Benchmark",
        {
            "Questions": (
                f"Old global Visual-JEPA comparison: `{best('current_visual_jepa')}`\n\n"
                f"Dense latent error: `{best('dense_visual_jepa_latent_error')}`\n\n"
                f"Dense kNN/PatchCore-lite: `{best('dense_visual_jepa_knn')}`\n\n"
                f"Dense PaDiM-lite: `{best('dense_visual_jepa_padim')}`\n\n"
                f"Pixel-stat: `{best('pixel_stat_baseline')}`\n\n"
                f"ResNet kNN/PatchCore-lite: `{best('resnet18_knn')}`\n\n"
                f"DINO: `{best('dinov2')}`"
            ),
            "Ranking": markdown_table(ranking.head(25).to_dict("records")),
            "Conclusion Rules": (
                "No SOTA claim. PatchCore-lite and PaDiM-lite are lightweight baselines. "
                "DINOv3 is pending unless official weights are actually available. "
                f"Elapsed seconds: `{time.time() - start:.2f}`."
            ),
        },
    )
    return {
        "results": results_path,
        "mean_std": mean_std_path,
        "per_category": per_category_path,
        "ranking": ranking_path,
        "report": out_root / "report.md",
    }
