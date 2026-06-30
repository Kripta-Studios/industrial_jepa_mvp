from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from common.config import load_config
from sensor_jepa.train.pretrain import pretrain_sensor_jepa


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sensor_jepa/demo_sensor_quick.yaml")
    parser.add_argument("--force-data", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    ckpt, history = pretrain_sensor_jepa(cfg, force_data=args.force_data)
    print(f"Saved {ckpt}; final_loss={history[-1]['loss']:.6f}")


if __name__ == "__main__":
    main()

