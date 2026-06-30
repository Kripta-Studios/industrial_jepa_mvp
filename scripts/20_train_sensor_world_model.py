from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from common.config import load_config
from sensor_jepa.train.world_model import evaluate_sensor_world_model, pretrain_sensor_world_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sensor_jepa/demo_sensor_quick.yaml")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
    ckpt, history, bundle = pretrain_sensor_world_model(cfg)
    metrics = evaluate_sensor_world_model(cfg, bundle)
    print(f"Saved {ckpt}; final_loss={history[-1]['loss']:.6f}")
    print(metrics)


if __name__ == "__main__":
    main()
