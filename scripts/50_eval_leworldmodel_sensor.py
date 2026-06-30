from __future__ import annotations

import argparse
from pathlib import Path

import _iwm_bootstrap  # noqa: F401


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LeWorldModel sensor smoke outputs.")
    parser.add_argument("--config", default="configs/industrial_world_model/leworldmodel_cnc.yaml")
    parser.add_argument("--out-root", default="outputs/industrial_world_model/leworldmodel")
    args = parser.parse_args()
    report = Path(args.out_root) / "report.md"
    if not report.exists():
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text("# LeWorldModel Evaluation\n\nStatus: missing pretrain output.\n", encoding="utf-8")
    print(f"LeWorldModel report: {report}")


if __name__ == "__main__":
    main()
