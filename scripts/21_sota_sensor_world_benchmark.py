from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from common.config import load_config
from sensor_jepa.train.sota_benchmark import run_sota_benchmark


def _parse_ints(value: str) -> list[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sensor_jepa/demo_sensor_quick.yaml")
    parser.add_argument("--seeds", default="42,123,999")
    parser.add_argument("--horizons", default="1,3,5,10,20")
    parser.add_argument("--targets", default="5,10,20")
    parser.add_argument("--out-root", default="outputs/sensor_jepa/sota_benchmark")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    seeds = _parse_ints(args.seeds)
    horizons = _parse_ints(args.horizons)
    targets = _parse_ints(args.targets)
    if args.quick:
        seeds = seeds[:1]
        horizons = horizons[:2]
        targets = targets[:2]
    paths = run_sota_benchmark(cfg, seeds=seeds, horizons=horizons, targets=targets, out_root=args.out_root, quick=args.quick)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

