from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from common.config import load_config
from visual_jepa.train.dense_benchmark import run_dense_visual_benchmark


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/visual_jepa/dense_visual_benchmark.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    paths = run_dense_visual_benchmark(cfg)
    for key, value in paths.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
