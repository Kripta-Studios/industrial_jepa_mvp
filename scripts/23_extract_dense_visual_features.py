from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from common.config import load_config
from visual_jepa.train.extract_dense_features import extract_dense_features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/visual_jepa/dense_mvtec_bottle_quick.yaml")
    parser.add_argument("--backbone", default="dense_visual_jepa")
    args = parser.parse_args()
    cfg = load_config(args.config)
    out = extract_dense_features(cfg, backbone=args.backbone)
    print(f"manifest: {out['manifest']}")
    print(f"backbone_info: {out['backbone_info']}")
    for path in out["paths"][:10]:
        print(f"features: {path}")


if __name__ == "__main__":
    main()
