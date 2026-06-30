from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from common.config import load_config
from visual_jepa.data.mvtec_ad import prepare_mvtec_from_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/visual_jepa/demo_visual_quick.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    bundle = prepare_mvtec_from_config(cfg, force=args.force)
    print(f"Prepared MVTec manifest: train={len(bundle.train_dataset)}, test={len(bundle.test_dataset)}")


if __name__ == "__main__":
    main()

