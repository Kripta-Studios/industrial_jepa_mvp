from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from common.config import load_config
from sensor_jepa.eval.dense_sensor_surprise import evaluate_dense_sensor_surprise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sensor_jepa/dense_sensor_cnc.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    paths = evaluate_dense_sensor_surprise(cfg)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
