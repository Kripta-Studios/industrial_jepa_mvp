from __future__ import annotations

import argparse
from pathlib import Path

import _iwm_bootstrap  # noqa: F401


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LeJEPA visual outputs.")
    parser.add_argument("--config", default="configs/industrial_world_model/lejepa_visual_mvtec.yaml")
    parser.add_argument("--out-root", default="outputs/industrial_world_model/lejepa_visual")
    args = parser.parse_args()
    out = Path(args.out_root)
    report = out / "report.md"
    if not report.exists():
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text("# LeJEPA Visual Evaluation\n\nStatus: missing pretrain output.\n", encoding="utf-8")
    print(f"LeJEPA visual report: {report}")


if __name__ == "__main__":
    main()
