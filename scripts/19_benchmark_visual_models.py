from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from common.config import load_config
from visual_jepa.train.evaluate import evaluate_visual_jepa
from visual_jepa.train.pretrain import pretrain_visual_jepa


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/visual_jepa/demo_visual_quick.yaml")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if not Path(cfg["outputs"]["checkpoint"]).exists():
        pretrain_visual_jepa(cfg)
    rows = evaluate_visual_jepa(cfg, include_baseline=True)
    print(rows)
    print("PatchCore/PaDiM are pending because anomalib is not installed in this environment.")


if __name__ == "__main__":
    main()

