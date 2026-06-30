from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_hierarchy_report(out_dir: str | Path, results: pd.DataFrame) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / "results.csv", index=False)
    results.head(10).to_csv(out_dir / "top_alerts.csv", index=False)
    (out_dir / "report.md").write_text(
        "# Hierarchical Industrial World Model Report\n\n"
        "This report aggregates patch/window anomaly scores into interpretable industrial alert rankings.\n\n"
        f"- Items scored: {len(results)}\n"
        "- Status: MVP aggregation; group labels and lot/cycle metadata improve usefulness when provided by a client.\n",
        encoding="utf-8",
    )
