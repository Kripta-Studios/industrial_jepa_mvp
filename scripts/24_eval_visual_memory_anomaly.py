from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from common.config import load_config
from visual_jepa.train.memory_anomaly import evaluate_visual_memory_anomaly


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/visual_jepa/dense_mvtec_bottle_quick.yaml")
    parser.add_argument("--backbone", default="dense_visual_jepa")
    args = parser.parse_args()
    cfg = load_config(args.config)
    paths = evaluate_visual_memory_anomaly(cfg, backbone=args.backbone)
    for key, value in paths.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
