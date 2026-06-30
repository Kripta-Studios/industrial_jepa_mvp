from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_world_model_report(out_dir: str | Path, logs: dict) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([logs]).to_csv(out_dir / "surprise_results.csv", index=False)
    (out_dir / "report.md").write_text(
        "# LeWorldModel Smoke Report\n\n"
        f"- Prediction loss: {logs.get('prediction_loss')}\n"
        f"- SIGReg loss: {logs.get('sigreg_loss')}\n"
        f"- Total loss: {logs.get('loss')}\n"
        f"- Surprise mean: {logs.get('surprise_mean')}\n"
        f"- Surprise max: {logs.get('surprise_max')}\n"
        f"- Surprise top-k mean: {logs.get('surprise_topk_mean')}\n"
        f"- Surprise EWMA last: {logs.get('surprise_ewma_last')}\n"
        f"- Collapse flag: {logs.get('collapse_flag')}\n"
        "- Status: synthetic smoke validation. Real predictive quality requires temporal process data and actions/setpoints.\n",
        encoding="utf-8",
    )
