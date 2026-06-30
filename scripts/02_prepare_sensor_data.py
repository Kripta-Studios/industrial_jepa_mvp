from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from common.config import load_config
from sensor_jepa.data.cnc_milling import prepare_from_config, save_bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sensor_jepa/demo_sensor_quick.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    bundle = prepare_from_config(cfg, force=args.force)
    print(f"Prepared CNC windows: train={bundle.x_train.shape}, val={bundle.x_val.shape}, test={bundle.x_test.shape}")


if __name__ == "__main__":
    main()

