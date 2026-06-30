from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from common.reports import markdown_table, write_markdown_report


def summarize_sensor_outputs(cfg: dict[str, Any]) -> Path:
    root = Path(cfg["outputs"]["root"])
    rows = []
    for path in root.rglob("*.csv"):
        if path.name in {"pretrain_history.csv", "world_model_history.csv"} or path.name.endswith("_scores.csv"):
            continue
        try:
            df = pd.read_csv(path)
            if len(df) > 20:
                continue
            rows.extend(df.to_dict("records"))
        except Exception:
            pass
    report = root / "reports" / "sensor_experiment_summary.md"
    write_markdown_report(
        report,
        "Sensor Experiment Summary",
        {
            "Runs": markdown_table(rows),
            "SOTA Notice": "These are MVP results. They are not SOTA claims.",
        },
    )
    return report
