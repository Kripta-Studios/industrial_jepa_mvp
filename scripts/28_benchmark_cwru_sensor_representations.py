from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from common.config import load_config
from common.paths import ensure_dir
from common.reports import write_markdown_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sensor_jepa/cwru_bearing.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    out_dir = ensure_dir(cfg.get("outputs", {}).get("root", "outputs/sensor_jepa/cwru_benchmark"))
    report = Path(out_dir) / "report.md"
    write_markdown_report(
        report,
        "CWRU Sensor Representation Benchmark",
        {
            "Status": "pending",
            "Reason": "External validation is planned after incremental CNC and DenseSensorJEPA MVP checks.",
        },
    )
    print(f"report: {report}")


if __name__ == "__main__":
    main()
