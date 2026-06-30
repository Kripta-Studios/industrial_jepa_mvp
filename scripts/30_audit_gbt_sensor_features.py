from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from common.config import load_config
from sensor_jepa.train.gbt_audit import run_gbt_audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sensor_jepa/incremental_value_cnc.yaml")
    parser.add_argument("--out-root", default="outputs/sensor_jepa/incremental_value_benchmark")
    args = parser.parse_args()
    cfg = load_config(args.config)
    paths = run_gbt_audit(cfg, out_dir=args.out_root)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
