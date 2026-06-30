from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any
import json

import numpy as np
import pandas as pd

from common.config import get_device_name
from common.paths import ensure_dir
from common.reports import markdown_table, write_markdown_report
from common.seed import seed_everything
from sensor_jepa.data.cnc_world_model import TransitionBundle, prepare_transition_from_config
from sensor_jepa.eval.incremental_metrics import add_delta_vs_metadata
from sensor_jepa.train.incremental_value_benchmark import (
    _fit_scores,
    _flatten,
    _metadata_matrix,
    _row_from_scores,
    _world_features,
    engineered_sensor_features,
)
from sensor_jepa.train.sota_benchmark import leakage_report


CORE_MODELS = [
    "metadata_only_no_cycle",
    "sensor_raw_only",
    "sensor_engineered_only",
    "current_z_only",
    "predicted_future_z_only",
    "metadata_plus_current_z",
    "metadata_plus_predicted_future_z",
    "metadata_plus_current_z_plus_predicted_future_z",
    "sensor_raw_plus_current_z",
    "sensor_raw_plus_predicted_future_z",
    "sensor_raw_plus_current_z_plus_predicted_future_z",
    "metadata_plus_sensor_raw",
    "metadata_plus_sensor_raw_plus_current_z",
]


def _cfg_for_combo(cfg: dict[str, Any], seed: int, horizon: int, target: int, out_root: Path) -> dict[str, Any]:
    combo = deepcopy(cfg)
    combo["seed"] = seed
    combo.setdefault("world_model", {})
    combo["world_model"]["forecast_horizon"] = horizon
    combo["world_model"]["failure_horizon_cycles"] = target
    combo.setdefault("outputs", {})
    combo["outputs"]["root"] = str(out_root / f"seed_{seed}" / f"h_{horizon}" / f"k_{target}")
    return combo


def _add_feature_row(
    rows: list[dict[str, Any]],
    tool_rows: list[dict[str, Any]],
    bundle: TransitionBundle,
    model_name: str,
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    feature_groups: str,
    seed: int,
    horizon: int,
    target: int,
    estimator: str,
) -> None:
    val_scores, test_scores, _, train_sec, notes = _fit_scores(x_train, bundle.y_failure_train, x_val, x_test, estimator, seed)
    row, by_tool = _row_from_scores(
        bundle,
        model_name=model_name,
        model_family="no_cycle_evidence",
        protocol="no_cycle",
        feature_groups=feature_groups,
        estimator=estimator,
        val_scores=val_scores,
        test_scores=test_scores,
        seed=seed,
        horizon=horizon,
        target=target,
        train_time_sec=train_sec,
        notes=notes,
    )
    rows.append(row)
    tool_rows.extend(by_tool)


def _mean_std(results: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        c
        for c in [
            "AUROC",
            "AUPRC",
            "precision_at_10pct",
            "recall_at_10pct",
            "F1",
            "balanced_accuracy",
            "false_alarms_per_tool",
            "mean_lead_time",
            "median_lead_time",
            "brier_score",
            "ECE",
            "delta_AUPRC_vs_metadata_only",
            "delta_AUROC_vs_metadata_only",
            "delta_Precision@10_vs_metadata_only",
            "delta_Recall@10_vs_metadata_only",
            "delta_lead_time_vs_metadata_only",
        ]
        if c in results
    ]
    groups = ["model_name", "forecast_horizon", "failure_horizon_cycles"]
    if results.empty:
        return pd.DataFrame()
    out = results.groupby(groups, dropna=False)[metrics].agg(["mean", "std", "count"]).reset_index()
    out.columns = ["_".join(str(part) for part in col if part) for col in out.columns.to_flat_index()]
    return out.sort_values(["AUPRC_mean", "AUROC_mean"], ascending=False).reset_index(drop=True)


def _horizon_target_summary(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    metrics = [c for c in ["AUROC", "AUPRC", "precision_at_10pct", "recall_at_10pct", "delta_AUPRC_vs_metadata_only"] if c in results]
    return (
        results.groupby(["model_name", "forecast_horizon", "failure_horizon_cycles"], dropna=False)[metrics]
        .mean()
        .reset_index()
        .sort_values(["forecast_horizon", "failure_horizon_cycles", "AUPRC"], ascending=[True, True, False])
    )


def _comparison_delta(
    df: pd.DataFrame,
    group_cols: list[str],
    baseline_model: str,
    candidate_model: str,
    delta_prefix: str,
) -> pd.DataFrame:
    base = df[df["model_name"].eq(baseline_model)]
    cand = df[df["model_name"].eq(candidate_model)]
    metrics = ["AUPRC", "precision_at_10pct", "recall_at_10pct", "mean_lead_time"]
    keep = group_cols + metrics
    if base.empty or cand.empty:
        return pd.DataFrame()
    merged = cand[keep].merge(base[keep], on=group_cols, suffixes=("_candidate", "_baseline"))
    rows = []
    for _, row in merged.iterrows():
        out = {c: row[c] for c in group_cols}
        out.update({"baseline_model": baseline_model, "candidate_model": candidate_model})
        for metric in metrics:
            out[f"delta_{metric}_{delta_prefix}"] = row[f"{metric}_candidate"] - row[f"{metric}_baseline"]
        rows.append(out)
    return pd.DataFrame(rows)


def sensor_vs_jepa_value(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_cols = [c for c in ["split_name", "seed", "forecast_horizon", "failure_horizon_cycles"] if c in results.columns]
    comparisons = [
        ("sensor_raw_only", "current_z_only", "current_z_vs_raw"),
        ("sensor_raw_only", "predicted_future_z_only", "future_z_vs_raw"),
        ("sensor_raw_only", "sensor_raw_plus_current_z", "raw_plus_z_vs_raw"),
        ("sensor_raw_only", "sensor_raw_plus_predicted_future_z", "raw_plus_future_z_vs_raw"),
        ("sensor_raw_only", "sensor_raw_plus_current_z_plus_predicted_future_z", "raw_plus_z_future_vs_raw"),
        ("metadata_plus_sensor_raw", "metadata_plus_sensor_raw_plus_current_z", "metadata_raw_z_vs_metadata_raw"),
    ]
    pieces = [_comparison_delta(results, group_cols, base, cand, prefix) for base, cand, prefix in comparisons]
    detail = pd.concat([p for p in pieces if not p.empty], ignore_index=True) if any(not p.empty for p in pieces) else pd.DataFrame()
    if detail.empty:
        return detail, pd.DataFrame()
    delta_cols = [c for c in detail.columns if c.startswith("delta_")]
    summary = detail.groupby(["baseline_model", "candidate_model"], dropna=False)[delta_cols].agg(["mean", "std", "count"]).reset_index()
    summary.columns = ["_".join(str(part) for part in col if part) for col in summary.columns.to_flat_index()]
    return detail, summary


def run_no_cycle_evidence_benchmark(
    cfg: dict[str, Any],
    seeds: list[int] | None = None,
    horizons: list[int] | None = None,
    targets: list[int] | None = None,
    out_root: str | Path = "outputs/sensor_jepa/no_cycle_evidence",
) -> dict[str, Path]:
    out_root = ensure_dir(out_root)
    bench_cfg = cfg.get("no_cycle_evidence", {})
    seeds = seeds or [int(s) for s in bench_cfg.get("seeds", [42, 123, 999])]
    horizons = horizons or [int(h) for h in bench_cfg.get("horizons", [1, 3, 5])]
    targets = targets or [int(k) for k in bench_cfg.get("targets", [5, 10, 20])]
    estimator = bench_cfg.get("estimator", "logistic_regression")
    world_epochs = int(bench_cfg.get("world_model_epochs", cfg.get("world_model", {}).get("epochs", 2)))
    device = get_device_name(cfg.get("device", "auto"))
    rows: list[dict[str, Any]] = []
    tool_rows: list[dict[str, Any]] = []
    leakage_checks: list[dict[str, Any]] = []

    for seed in seeds:
        for horizon in horizons:
            for target in targets:
                seed_everything(seed)
                combo_cfg = _cfg_for_combo(cfg, seed, horizon, target, out_root)
                combo_cfg.setdefault("world_model", {})
                combo_cfg["world_model"]["epochs"] = world_epochs
                bundle = prepare_transition_from_config(combo_cfg)
                leakage_checks.append({"seed": seed, "forecast_horizon": horizon, "failure_horizon_cycles": target, **leakage_report(bundle)})
                meta_train, meta_names = _metadata_matrix(bundle, "train", include_cycle=False)
                meta_val, _ = _metadata_matrix(bundle, "val", include_cycle=False)
                meta_test, _ = _metadata_matrix(bundle, "test", include_cycle=False)
                eng_train, eng_names = engineered_sensor_features(bundle.x_train)
                eng_val, _ = engineered_sensor_features(bundle.x_val)
                eng_test, _ = engineered_sensor_features(bundle.x_test)
                raw_train = _flatten(bundle.x_train)
                raw_val = _flatten(bundle.x_val)
                raw_test = _flatten(bundle.x_test)
                world = _world_features(combo_cfg, bundle, device)
                z_train, z_val, z_test, z_names = world["current_z"]
                pf_train, pf_val, pf_test, pf_names = world["predicted_future_z"]

                feature_sets = [
                    ("metadata_only_no_cycle", meta_train, meta_val, meta_test, "metadata_features_no_cycle"),
                    ("sensor_raw_only", raw_train, raw_val, raw_test, "sensor_raw_features"),
                    ("sensor_engineered_only", eng_train, eng_val, eng_test, "sensor_engineered_features"),
                    ("current_z_only", z_train, z_val, z_test, "jepa_global_embeddings"),
                    ("predicted_future_z_only", pf_train, pf_val, pf_test, "world_model_features"),
                    ("metadata_plus_current_z", np.concatenate([meta_train, z_train], axis=1), np.concatenate([meta_val, z_val], axis=1), np.concatenate([meta_test, z_test], axis=1), "metadata_features_no_cycle,jepa_global_embeddings"),
                    ("metadata_plus_predicted_future_z", np.concatenate([meta_train, pf_train], axis=1), np.concatenate([meta_val, pf_val], axis=1), np.concatenate([meta_test, pf_test], axis=1), "metadata_features_no_cycle,world_model_features"),
                    ("metadata_plus_current_z_plus_predicted_future_z", np.concatenate([meta_train, z_train, pf_train], axis=1), np.concatenate([meta_val, z_val, pf_val], axis=1), np.concatenate([meta_test, z_test, pf_test], axis=1), "metadata_features_no_cycle,jepa_global_embeddings,world_model_features"),
                    ("sensor_raw_plus_current_z", np.concatenate([raw_train, z_train], axis=1), np.concatenate([raw_val, z_val], axis=1), np.concatenate([raw_test, z_test], axis=1), "sensor_raw_features,jepa_global_embeddings"),
                    ("sensor_raw_plus_predicted_future_z", np.concatenate([raw_train, pf_train], axis=1), np.concatenate([raw_val, pf_val], axis=1), np.concatenate([raw_test, pf_test], axis=1), "sensor_raw_features,world_model_features"),
                    ("sensor_raw_plus_current_z_plus_predicted_future_z", np.concatenate([raw_train, z_train, pf_train], axis=1), np.concatenate([raw_val, z_val, pf_val], axis=1), np.concatenate([raw_test, z_test, pf_test], axis=1), "sensor_raw_features,jepa_global_embeddings,world_model_features"),
                    ("metadata_plus_sensor_raw", np.concatenate([meta_train, raw_train], axis=1), np.concatenate([meta_val, raw_val], axis=1), np.concatenate([meta_test, raw_test], axis=1), "metadata_features_no_cycle,sensor_raw_features"),
                    ("metadata_plus_sensor_raw_plus_current_z", np.concatenate([meta_train, raw_train, z_train], axis=1), np.concatenate([meta_val, raw_val, z_val], axis=1), np.concatenate([meta_test, raw_test, z_test], axis=1), "metadata_features_no_cycle,sensor_raw_features,jepa_global_embeddings"),
                ]
                for model_name, x_train, x_val, x_test, groups in feature_sets:
                    _add_feature_row(rows, tool_rows, bundle, model_name, x_train, x_val, x_test, groups, seed, horizon, target, estimator)

    results = pd.DataFrame(rows)
    results = add_delta_vs_metadata(
        results,
        baseline_model="metadata_only_no_cycle",
        group_columns=["protocol", "seed", "forecast_horizon", "failure_horizon_cycles"],
    )
    by_seed = results.copy()
    mean_std = _mean_std(results)
    by_horizon = _horizon_target_summary(results)
    value_detail, value_summary = sensor_vs_jepa_value(results)
    leakage = {"checks": leakage_checks, "passes": all(check["passes"] for check in leakage_checks)}

    paths = {
        "results": out_root / "results.csv",
        "results_mean_std": out_root / "results_mean_std.csv",
        "results_by_horizon_target": out_root / "results_by_horizon_target.csv",
        "results_by_seed": out_root / "results_by_seed.csv",
        "results_by_tool": out_root / "results_by_tool.csv",
        "leakage_report": out_root / "leakage_report.json",
        "report": out_root / "report.md",
        "sensor_vs_jepa_detail": Path("outputs/sensor_jepa/sensor_vs_jepa_value/no_cycle_detail.csv"),
        "sensor_vs_jepa_summary": Path("outputs/sensor_jepa/sensor_vs_jepa_value/no_cycle_summary.csv"),
    }
    results.to_csv(paths["results"], index=False)
    mean_std.to_csv(paths["results_mean_std"], index=False)
    by_horizon.to_csv(paths["results_by_horizon_target"], index=False)
    by_seed.to_csv(paths["results_by_seed"], index=False)
    pd.DataFrame(tool_rows).to_csv(paths["results_by_tool"], index=False)
    paths["leakage_report"].write_text(json.dumps(leakage, indent=2), encoding="utf-8")
    ensure_dir(paths["sensor_vs_jepa_detail"].parent)
    value_detail.to_csv(paths["sensor_vs_jepa_detail"], index=False)
    value_summary.to_csv(paths["sensor_vs_jepa_summary"], index=False)

    top = mean_std.head(20)
    raw_vs = value_summary.head(20) if not value_summary.empty else pd.DataFrame()
    def _best_model(name: str) -> str:
        row = mean_std[mean_std["model_name"].eq(name)]
        if row.empty:
            return "missing"
        best = row.sort_values("AUPRC_mean", ascending=False).iloc[0]
        delta = best.get("delta_AUPRC_vs_metadata_only_mean", np.nan)
        return f"AUPRC={best['AUPRC_mean']:.4f} +/- {best['AUPRC_std']:.4f}; delta={delta:+.4f}"

    write_markdown_report(
        paths["report"],
        "No-Cycle Sensor Evidence Benchmark",
        {
            "Protocol": "No explicit cycle/lifecycle proxies. Metadata rows use action/process context only.",
            "Leakage": f"`{leakage['passes']}`",
            "Metadata No-Cycle": _best_model("metadata_only_no_cycle"),
            "Sensor Raw": _best_model("sensor_raw_only"),
            "Current Z": _best_model("current_z_only"),
            "Predicted Future Z": _best_model("predicted_future_z_only"),
            "Raw vs JEPA Deltas": markdown_table(raw_vs.to_dict("records")) if len(raw_vs) else "No raw-vs-JEPA comparison rows.",
            "Top Mean/Std Rows": markdown_table(top.to_dict("records")) if len(top) else "No rows.",
            "Conclusion Rule": "Use deltas within the same seed/h/K. Do not compare best h/K across models as final evidence.",
        },
    )
    return paths
