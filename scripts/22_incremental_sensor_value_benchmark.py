from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from common.config import load_config
from sensor_jepa.train.incremental_value_benchmark import run_incremental_value_benchmark


def _parse_ints(value: str | None) -> list[int] | None:
    if value is None:
        return None
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sensor_jepa/incremental_value_cnc.yaml")
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--horizons", default=None)
    parser.add_argument("--targets", default=None)
    parser.add_argument("--out-root", default=None)
    parser.add_argument("--world-model-epochs", type=int, default=None)
    parser.add_argument("--no-gbt", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg.setdefault("incremental_benchmark", {})
    seeds = _parse_ints(args.seeds)
    horizons = _parse_ints(args.horizons)
    targets = _parse_ints(args.targets)
    if seeds is not None:
        cfg["incremental_benchmark"]["seeds"] = seeds
    if horizons is not None:
        cfg["incremental_benchmark"]["horizons"] = horizons
    if targets is not None:
        cfg["incremental_benchmark"]["targets"] = targets
    if args.out_root is not None:
        cfg["incremental_benchmark"]["output_dir"] = args.out_root
    if args.world_model_epochs is not None:
        cfg["incremental_benchmark"]["world_model_epochs"] = args.world_model_epochs
    if args.no_gbt:
        cfg["incremental_benchmark"]["include_gbt_feature_value"] = False
    paths = run_incremental_value_benchmark(cfg)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
