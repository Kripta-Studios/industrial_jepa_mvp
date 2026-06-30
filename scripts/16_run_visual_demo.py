from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from common.config import load_config
from common.manifest import write_dataset_manifest
from visual_jepa.data.mvtec_ad import prepare_mvtec_from_config
from visual_jepa.train.evaluate import evaluate_visual_jepa
from visual_jepa.train.pretrain import pretrain_visual_jepa


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/visual_jepa/demo_visual_quick.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    write_dataset_manifest()
    bundle = prepare_mvtec_from_config(cfg, force=True)
    ckpt, _ = pretrain_visual_jepa(cfg, force_data=False)
    rows = evaluate_visual_jepa(cfg, include_baseline=True)
    print(f"Prepared train={len(bundle.train_dataset)} test={len(bundle.test_dataset)}")
    print(f"Saved {ckpt}")
    print(rows)


if __name__ == "__main__":
    main()

