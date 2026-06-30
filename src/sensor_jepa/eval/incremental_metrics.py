from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_GROUP_COLUMNS = ["protocol", "seed", "forecast_horizon", "failure_horizon_cycles"]

DELTA_METRICS = {
    "AUPRC": "delta_AUPRC_vs_metadata_only",
    "AUROC": "delta_AUROC_vs_metadata_only",
    "precision_at_10pct": "delta_Precision@10_vs_metadata_only",
    "recall_at_10pct": "delta_Recall@10_vs_metadata_only",
    "false_alarms_per_tool": "delta_false_alarms_vs_metadata_only",
    "mean_lead_time": "delta_lead_time_vs_metadata_only",
}


def existing_group_columns(df: pd.DataFrame, columns: Iterable[str] = DEFAULT_GROUP_COLUMNS) -> list[str]:
    return [c for c in columns if c in df.columns]


def validate_unique_baseline(
    results: pd.DataFrame,
    baseline_model: str = "metadata_only",
    group_columns: Iterable[str] = DEFAULT_GROUP_COLUMNS,
) -> None:
    """Ensure each comparison group has at most one metadata baseline row."""

    if results.empty or "model_name" not in results.columns:
        return
    group_cols = existing_group_columns(results, group_columns)
    if not group_cols:
        return
    baseline = results[results["model_name"].eq(baseline_model)]
    if baseline.empty:
        return
    dup = baseline.groupby(group_cols, dropna=False).size().reset_index(name="n")
    dup = dup[dup["n"] > 1]
    if not dup.empty:
        raise ValueError(
            "metadata baseline is not unique for comparison groups: "
            + dup[group_cols + ["n"]].to_dict("records").__repr__()
        )


def add_delta_vs_metadata(
    results: pd.DataFrame,
    baseline_model: str = "metadata_only",
    group_columns: Iterable[str] = DEFAULT_GROUP_COLUMNS,
) -> pd.DataFrame:
    """Add deltas against metadata-only within the same protocol/seed/h/K group."""

    if results.empty or "model_name" not in results.columns:
        return results.copy()
    out = results.copy()
    group_cols = existing_group_columns(out, group_columns)
    validate_unique_baseline(out, baseline_model=baseline_model, group_columns=group_cols)
    baseline_cols = group_cols + [m for m in DELTA_METRICS if m in out.columns]
    baseline = out[out["model_name"].eq(baseline_model)][baseline_cols].copy()
    if baseline.empty:
        for delta_col in DELTA_METRICS.values():
            out[delta_col] = np.nan
        return out
    rename = {m: f"{m}__metadata_baseline" for m in DELTA_METRICS if m in baseline.columns}
    baseline = baseline.rename(columns=rename)
    out = out.merge(baseline, on=group_cols, how="left")
    for metric, delta_col in DELTA_METRICS.items():
        baseline_metric = f"{metric}__metadata_baseline"
        if metric in out.columns and baseline_metric in out.columns:
            out[delta_col] = out[metric] - out[baseline_metric]
        else:
            out[delta_col] = np.nan
    return out


def summarize_by_protocol(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    metrics = [c for c in ["AUROC", "AUPRC", "precision_at_10pct", "recall_at_10pct", "ECE", "brier_score"] if c in results]
    group_cols = [c for c in ["protocol", "model_name", "model_family"] if c in results]
    if not metrics or not group_cols:
        return pd.DataFrame()
    agg = results.groupby(group_cols, dropna=False)[metrics].agg(["mean", "std"]).reset_index()
    agg.columns = ["_".join(str(part) for part in col if part) for col in agg.columns.to_flat_index()]
    sort_col = "AUPRC_mean" if "AUPRC_mean" in agg.columns else agg.columns[-1]
    return agg.sort_values(sort_col, ascending=False).reset_index(drop=True)


def summarize_by_horizon_target(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    group_cols = [c for c in ["protocol", "model_name", "forecast_horizon", "failure_horizon_cycles"] if c in results]
    metrics = [c for c in ["AUROC", "AUPRC", "precision_at_10pct", "recall_at_10pct"] if c in results]
    if not group_cols or not metrics:
        return pd.DataFrame()
    return results.groupby(group_cols, dropna=False)[metrics].mean().reset_index()


def feature_group_ablation(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty or "feature_groups" not in results:
        return pd.DataFrame()
    keep = [c for c in ["protocol", "model_name", "estimator", "feature_groups", "AUPRC", "AUROC", "precision_at_10pct", "recall_at_10pct"] if c in results]
    return results[keep].copy().sort_values([c for c in ["protocol", "AUPRC"] if c in keep], ascending=[True, False] if "AUPRC" in keep else True)
