from __future__ import annotations

import numpy as np
import pandas as pd


def aggregate_scores(scores: np.ndarray, method: str = "topk_mean", top_k: int = 5) -> float:
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        return 0.0
    if method == "mean":
        return float(scores.mean())
    if method == "max":
        return float(scores.max())
    if method == "ewma":
        alpha = 0.3
        value = scores[0]
        for s in scores[1:]:
            value = alpha * s + (1 - alpha) * value
        return float(value)
    k = max(1, min(int(top_k), len(scores)))
    return float(np.sort(scores)[-k:].mean())


def group_risk_table(df: pd.DataFrame, group_col: str, score_col: str, method: str = "topk_mean") -> pd.DataFrame:
    rows = []
    for group, part in df.groupby(group_col):
        rows.append({"group": group, "risk_score": aggregate_scores(part[score_col].to_numpy(), method=method), "count": len(part)})
    return pd.DataFrame(rows).sort_values("risk_score", ascending=False).reset_index(drop=True)


def top_alerts(df: pd.DataFrame, score_col: str = "risk_score", k: int = 10) -> pd.DataFrame:
    return df.sort_values(score_col, ascending=False).head(k).reset_index(drop=True)
