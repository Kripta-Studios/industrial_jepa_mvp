from __future__ import annotations

import argparse
from pathlib import Path

import _iwm_bootstrap  # noqa: F401


def main() -> None:
    parser = argparse.ArgumentParser(description="Check hierarchical IWM report.")
    parser.add_argument("--out-root", default="outputs/industrial_world_model/hierarchy")
    args = parser.parse_args()
    report = Path(args.out_root) / "report.md"
    if not report.exists():
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text("# Hierarchical IWM Report\n\nRun script 54 first.\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
