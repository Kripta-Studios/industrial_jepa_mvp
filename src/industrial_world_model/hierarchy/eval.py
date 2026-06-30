from __future__ import annotations

import pandas as pd

from .aggregation import top_alerts


def evaluate_hierarchy(df: pd.DataFrame, score_col: str = "risk_score") -> dict[str, float]:
    alerts = top_alerts(df, score_col=score_col, k=max(1, min(10, len(df))))
    return {"num_items": float(len(df)), "top_score": float(alerts[score_col].iloc[0]) if len(alerts) else 0.0}
