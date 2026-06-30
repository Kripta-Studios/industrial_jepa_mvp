from __future__ import annotations

from pathlib import Path

import pandas as pd

import _bootstrap  # noqa: F401
from common.reports import markdown_table, write_markdown_report


def main() -> None:
    root = Path("outputs/visual_jepa/demo_quick")
    rows = []
    for path in root.rglob("visual_benchmark_results.csv"):
        rows.extend(pd.read_csv(path).to_dict("records"))
    report = root / "reports" / "visual_experiment_summary.md"
    write_markdown_report(report, "Visual Experiment Summary", {"Runs": markdown_table(rows)})
    print(report)


if __name__ == "__main__":
    main()

