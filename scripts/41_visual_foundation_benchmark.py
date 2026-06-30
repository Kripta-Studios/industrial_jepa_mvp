from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import yaml

import _iwm_bootstrap  # noqa: F401

from industrial_world_model.visual.baselines import PixelStatBaseline
from industrial_world_model.visual.datasets import collect_mvtec_records, split_records
from industrial_world_model.visual.eval import image_anomaly_metrics
from industrial_world_model.visual.feature_extractors import build_feature_extractor
from industrial_world_model.visual.heatmaps import patch_scores_to_heatmap, save_heatmap_png
from industrial_world_model.visual.padim_lite import PadimLite
from industrial_world_model.visual.patchcore import PatchCoreLite
from industrial_world_model.visual.transforms import load_image_tensor


def _load_config(path: str | None) -> dict:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _stack(records, image_size: int) -> torch.Tensor:
    return torch.stack([load_image_tensor(r.image_path, image_size=image_size) for r in records])


def _select_categories(config: dict, cli_categories: str | None):
    cats = cli_categories or config.get("dataset", {}).get("categories", "bottle")
    if cats in (None, "all"):
        return None
    if isinstance(cats, str):
        return [c.strip() for c in cats.split(",") if c.strip()]
    return list(cats)


def _limit_quick_records(train_records, test_records, max_samples: int | None):
    if not max_samples:
        return train_records, test_records
    train_limit = max(4, max_samples // 2)
    test_limit = max(4, max_samples // 2)
    train_records = train_records[:train_limit]
    normal = [r for r in test_records if r.label == 0]
    anomaly = [r for r in test_records if r.label == 1]
    if normal and anomaly:
        half = max(1, test_limit // 2)
        test_records = normal[:half] + anomaly[: test_limit - half]
    else:
        test_records = test_records[:test_limit]
    return train_records, test_records


def run(config: dict, out_root: Path, quick: bool = False, categories: str | None = None, max_samples: int | None = None) -> pd.DataFrame:
    dataset_cfg = config.get("dataset", {})
    model_cfg = config.get("model", {})
    root = dataset_cfg.get("root", "data/raw/visual/mvtec_ad")
    dataset = dataset_cfg.get("name", "mvtec_ad")
    cats = _select_categories(config, categories)
    if quick:
        max_samples = max_samples or 120
    image_size = int(config.get("image_size", model_cfg.get("image_size", 224)))
    patch_size = int(model_cfg.get("patch_size", 16))
    backbone = model_cfg.get("backbone", "dinov3")

    records = collect_mvtec_records(root, dataset=dataset, categories=cats, max_samples=None)
    train_records, test_records = split_records(records)
    out_root.mkdir(parents=True, exist_ok=True)
    if not train_records or not test_records:
        row = {"dataset": dataset, "category": ",".join(cats or []), "model": "missing_dataset", "notes": "missing train/test records"}
        df = pd.DataFrame([row])
        df.to_csv(out_root / "results.csv", index=False)
        (out_root / "report.md").write_text("# Visual Foundation Report\n\nDataset missing or split not inferred.\n", encoding="utf-8")
        return df

    # Keep quick runs bounded while preserving both normal and anomalous test samples.
    train_records, test_records = _limit_quick_records(train_records, test_records, max_samples)

    train_x = _stack(train_records, image_size)
    test_x = _stack(test_records, image_size)
    y_true = torch.tensor([r.label for r in test_records]).numpy()
    category = train_records[0].category if train_records else (cats[0] if cats else "unknown")

    rows: list[dict] = []
    pixel = PixelStatBaseline().fit(train_x)
    pixel_scores, _ = pixel.score(test_x)
    rows.append({"dataset": dataset, "category": category, "model": "pixel_stat_baseline", **image_anomaly_metrics(y_true, pixel_scores.numpy()), "notes": "simple baseline"})

    extractor = build_feature_extractor(backbone, patch_size=patch_size, image_size=image_size)
    train_emb, grid_shape = extractor.extract(train_x)
    test_emb, _ = extractor.extract(test_x)

    patchcore = PatchCoreLite(coreset_ratio=float(model_cfg.get("coreset_ratio", 0.2)), top_k=int(model_cfg.get("top_k", 5)))
    pc_scores = patchcore.fit_score(train_emb, test_emb)
    rows.append({"dataset": dataset, "category": category, "model": f"{backbone}_patchcore_lite", **image_anomaly_metrics(y_true, pc_scores.image_scores.detach().numpy()), "notes": extractor.info.notes})

    if PadimLite is not None:
        padim = PadimLite(n_features=min(64, train_emb.shape[-1]), top_k=int(model_cfg.get("top_k", 5)))
        pd_scores = padim.fit_score(train_emb, test_emb)
        rows.append({"dataset": dataset, "category": category, "model": f"{backbone}_padim_lite", **image_anomaly_metrics(y_true, pd_scores.image_scores.detach().numpy()), "notes": extractor.info.notes})
        heat_dir = out_root / "heatmaps" / dataset / category
        heat_dir.mkdir(parents=True, exist_ok=True)
        for i in range(min(3, len(test_records))):
            heat = patch_scores_to_heatmap(pd_scores.patch_scores[i : i + 1], grid_shape, (image_size, image_size))[0]
            save_heatmap_png(heat, str(heat_dir / f"sample_{i:03d}.png"))

    df = pd.DataFrame(rows)
    df.to_csv(out_root / "results.csv", index=False)
    df.groupby(["dataset", "model"], dropna=False).mean(numeric_only=True).reset_index().to_csv(out_root / "results_mean_std.csv", index=False)
    df.to_csv(out_root / "results_by_dataset_category.csv", index=False)
    (out_root / "backbone_info.json").write_text(json.dumps(extractor.info.__dict__, indent=2), encoding="utf-8")
    (out_root / "report.md").write_text(
        "# Visual Foundation Report\n\n"
        f"- Dataset: {dataset}\n"
        f"- Category: {category}\n"
        f"- Requested backbone: {extractor.info.requested_backbone}\n"
        f"- Actual backbone: {extractor.info.actual_backbone}\n"
        f"- DINOv3 available: {extractor.info.dinov3_available}\n"
        f"- Rows: {len(df)}\n\n"
        "This benchmark separates simple visual baselines from dense-feature PatchCore/PaDiM scoring. "
        "DINOv3 is reported only when weights are actually loaded; otherwise the fallback is explicit.\n",
        encoding="utf-8",
    )
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run IWM visual foundation benchmark.")
    parser.add_argument("--config", default="configs/industrial_world_model/visual_foundation_mvtec.yaml")
    parser.add_argument("--out-root", default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--categories", default=None)
    args = parser.parse_args()
    cfg = _load_config(args.config)
    out = Path(args.out_root or cfg.get("output_dir", "outputs/industrial_world_model/visual_foundation"))
    run(cfg, out, quick=args.quick, categories=args.categories, max_samples=args.max_samples)
    print(f"Visual foundation results written to {out}")


if __name__ == "__main__":
    main()
