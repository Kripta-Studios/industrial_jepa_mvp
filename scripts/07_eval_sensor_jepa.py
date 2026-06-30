from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from common.config import load_config
from sensor_jepa.train.evaluate import summarize_sensor_outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sensor_jepa/demo_sensor_quick.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    report = summarize_sensor_outputs(cfg)
    print(f"Wrote {report}")


if __name__ == "__main__":
    main()

