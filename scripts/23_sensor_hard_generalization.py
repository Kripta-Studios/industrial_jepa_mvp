from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from common.config import load_config
from sensor_jepa.train.hard_generalization import run_hard_generalization


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sensor_jepa/hard_generalization_cnc.yaml")
    parser.add_argument("--out-root", default=None)
    parser.add_argument("--world-model-epochs", type=int, default=None)
    parser.add_argument("--no-latents", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg.setdefault("hard_generalization", {})
    if args.out_root is not None:
        cfg["hard_generalization"]["output_dir"] = args.out_root
    if args.world_model_epochs is not None:
        cfg["hard_generalization"]["world_model_epochs"] = args.world_model_epochs
    if args.no_latents:
        cfg["hard_generalization"]["include_latents"] = False
    paths = run_hard_generalization(cfg)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
