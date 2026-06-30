from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from common.config import load_config
from visual_jepa.train.evaluate import evaluate_visual_jepa


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/visual_jepa/demo_visual_quick.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    rows = evaluate_visual_jepa(cfg, include_baseline=False)
    print(rows)


if __name__ == "__main__":
    main()

