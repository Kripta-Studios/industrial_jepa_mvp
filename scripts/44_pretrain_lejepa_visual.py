from __future__ import annotations

import argparse
from pathlib import Path

import _iwm_bootstrap  # noqa: F401

from industrial_world_model.lejepa.eval import write_lejepa_report
from industrial_world_model.lejepa.train import train_lejepa_smoke


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke pretrain LeJEPA/SIGReg visual encoder.")
    parser.add_argument("--config", default="configs/industrial_world_model/lejepa_visual_mvtec.yaml")
    parser.add_argument("--out-root", default="outputs/industrial_world_model/lejepa_visual")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--categories", default="bottle")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    out = Path(args.out_root)
    logs = train_lejepa_smoke(out, epochs=args.epochs if args.quick else max(args.epochs, 2))
    write_lejepa_report(out, logs)
    print(f"LeJEPA visual smoke written to {out}")


if __name__ == "__main__":
    main()
