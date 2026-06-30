from __future__ import annotations

import argparse
from pathlib import Path

import _iwm_bootstrap  # noqa: F401

from industrial_world_model.lejepa.eval import write_lejepa_report
from industrial_world_model.lejepa.train import train_lejepa_smoke


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke pretrain LeJEPA/SIGReg sensor encoder.")
    parser.add_argument("--config", default="configs/industrial_world_model/lejepa_sensor_cnc.yaml")
    parser.add_argument("--out-root", default="outputs/industrial_world_model/sensor_lejepa")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()
    out = Path(args.out_root)
    logs = train_lejepa_smoke(out, epochs=args.epochs, input_dim=32, embedding_dim=32)
    write_lejepa_report(out, logs)
    print(f"Sensor LeJEPA smoke written to {out}")


if __name__ == "__main__":
    main()
