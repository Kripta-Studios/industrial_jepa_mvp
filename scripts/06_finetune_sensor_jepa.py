from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from common.config import load_config
from sensor_jepa.train.finetune import finetune_sensor_jepa


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sensor_jepa/demo_sensor_quick.yaml")
    parser.add_argument("--mode", default="full", choices=["frozen", "semi", "full"])
    parser.add_argument("--label-fraction", type=float, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    row = finetune_sensor_jepa(cfg, mode=args.mode, label_fraction=args.label_fraction)
    print(row)


if __name__ == "__main__":
    main()

