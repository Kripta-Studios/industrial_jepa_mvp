from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from common.config import load_config
from sensor_jepa.train.hard_generalization_evidence import run_hard_generalization_evidence


def _parse_ints(value: str | None) -> list[int] | None:
    if value is None:
        return None
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sensor_jepa/hard_generalization_cnc.yaml")
    parser.add_argument("--model-seeds", "--seeds", dest="model_seeds", default="42,123,999")
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--out-root", default="outputs/sensor_jepa/hard_generalization_evidence")
    args = parser.parse_args()
    cfg = load_config(args.config)
    paths = run_hard_generalization_evidence(
        cfg,
        model_seeds=_parse_ints(args.model_seeds),
        data_seed=args.data_seed,
        split_seed=args.split_seed,
        out_root=args.out_root,
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
