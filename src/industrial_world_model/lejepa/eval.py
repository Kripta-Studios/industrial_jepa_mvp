from __future__ import annotations

import pandas as pd
from pathlib import Path


def write_lejepa_report(out_dir: str | Path, logs: dict) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([logs]).to_csv(out_dir / "anomaly_results.csv", index=False)
    (out_dir / "report.md").write_text(
        "# LeJEPA/SIGReg Report\n\n"
        f"- Final loss: {logs.get('loss')}\n"
        f"- SIGReg: {logs.get('sigreg_loss')}\n"
        f"- Collapse flag: {logs.get('collapse_flag')}\n"
        "- Status: smoke implementation; downstream DINO comparison pending unless benchmark output exists.\n",
        encoding="utf-8",
    )
