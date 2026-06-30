from __future__ import annotations

import argparse
from pathlib import Path

import _iwm_bootstrap  # noqa: F401

from industrial_world_model.world_model.eval import write_world_model_report
from industrial_world_model.world_model.train import train_world_model_smoke


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain LeWorldModel sensor smoke.")
    parser.add_argument("--config", default="configs/industrial_world_model/leworldmodel_cnc.yaml")
    parser.add_argument("--out-root", default="outputs/industrial_world_model/leworldmodel")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lambda-sigreg", type=float, default=0.05)
    args = parser.parse_args()
    out = Path(args.out_root)
    logs = train_world_model_smoke(out, epochs=args.epochs, lambda_sigreg=args.lambda_sigreg)
    write_world_model_report(out, logs)
    print(f"LeWorldModel sensor smoke written to {out}")


if __name__ == "__main__":
    main()
