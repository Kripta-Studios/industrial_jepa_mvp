from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

import _iwm_bootstrap  # noqa: F401

from industrial_world_model.visual.datasets import collect_mvtec_records
from industrial_world_model.visual.feature_extractors import build_feature_extractor
from industrial_world_model.visual.transforms import load_image_tensor


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract IWM dense visual features.")
    parser.add_argument("--config", default="configs/industrial_world_model/visual_foundation_mvtec.yaml")
    parser.add_argument("--out-root", default="outputs/industrial_world_model/features")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--categories", default="bottle")
    parser.add_argument("--allow-fallback", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8")) or {}
    root = cfg.get("dataset", {}).get("root", "data/raw/visual/mvtec_ad")
    cats = [c.strip() for c in args.categories.split(",") if c.strip()]
    records = collect_mvtec_records(root, categories=cats, max_samples=args.max_samples if args.quick else None)
    out = Path(args.out_root)
    out.mkdir(parents=True, exist_ok=True)
    if not records:
        torch.save({"status": "missing", "features": torch.empty(0)}, out / "features.pt")
        return
    image_size = int(cfg.get("image_size", 224))
    extractor = build_feature_extractor(
        cfg.get("model", {}).get("backbone", "dinov3"),
        image_size=image_size,
        allow_fallback=args.allow_fallback or bool(cfg.get("model", {}).get("allow_fallback", False)),
    )
    x = torch.stack([load_image_tensor(r.image_path, image_size=image_size) for r in records])
    feats, grid = extractor.extract(x)
    torch.save({"features": feats, "grid_shape": grid, "records": [r.__dict__ for r in records], "backbone": extractor.info.__dict__}, out / "features.pt")
    print(f"Features written to {out / 'features.pt'}")


if __name__ == "__main__":
    main()
