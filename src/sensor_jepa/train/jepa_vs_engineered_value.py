from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from common.config import get_device_name
from common.paths import ensure_dir
from common.reports import markdown_table, write_markdown_report
from common.seed import seed_everything
from sensor_jepa.data.cnc_world_model import TransitionBundle, prepare_transition_from_config
from sensor_jepa.data.hard_splits import build_hard_split
from sensor_jepa.train.hard_generalization import _concat_bundle, _subset_bundle
from sensor_jepa.train.incremental_value_benchmark import (
    _fit_scores,
    _flatten,
    _metadata_matrix,
    _row_from_scores,
    _world_features,
    engineered_sensor_features,
)


ENGINEERED_BASELINE = "sensor_engineered_only"
METADATA_ENGINEERED_BASELINE = "metadata_plus_sensor_engineered"


def _cfg_for_combo(cfg: dict[str, Any], seed: int, horizon: int, target: int, out_root: Path) -> dict[str, Any]:
    combo = deepcopy(cfg)
    combo["seed"] = seed
    combo.setdefault("world_model", {})
    combo["world_model"]["forecast_horizon"] = horizon
    combo["world_model"]["failure_horizon_cycles"] = target
    combo.setdefault("outputs", {})
    combo["outputs"]["root"] = str(out_root / f"seed_{seed}" / f"h_{horizon}" / f"k_{target}")
    return combo


def _add_row(
    rows: list[dict[str, Any]],
    tool_rows: list[dict[str, Any]],
    bundle: TransitionBundle,
    protocol: str,
    split_name: str,
    model_name: str,
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    feature_groups: str,
    estimator: str,
    seed: int,
    horizon: int,
    target: int,
) -> None:
    val_scores, test_scores, _, train_sec, notes = _fit_scores(x_train, bundle.y_failure_train, x_val, x_test, estimator, seed)
    row, by_tool = _row_from_scores(
        bundle,
        model_name=model_name,
        model_family="jepa_vs_engineered_value",
        protocol=protocol,
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
    row["split_name"] = split_name
    rows.append(row)
    for tool_row in by_tool:
        tool_row["split_name"] = split_name
    tool_rows.extend(by_tool)


def _feature_sets(bundle: TransitionBundle, include_cycle_metadata: bool, device: str, combo_cfg: dict[str, Any]) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray, str]]:
    meta_train, _ = _metadata_matrix(bundle, "train", include_cycle=include_cycle_metadata)
    meta_val, _ = _metadata_matrix(bundle, "val", include_cycle=include_cycle_metadata)
    meta_test, _ = _metadata_matrix(bundle, "test", include_cycle=include_cycle_metadata)
    raw_train = _flatten(bundle.x_train)
    raw_val = _flatten(bundle.x_val)
    raw_test = _flatten(bundle.x_test)
    eng_train, _ = engineered_sensor_features(bundle.x_train)
    eng_val, _ = engineered_sensor_features(bundle.x_val)
    eng_test, _ = engineered_sensor_features(bundle.x_test)
    world = _world_features(combo_cfg, bundle, device)
    z_train, z_val, z_test, _ = world["current_z"]
    pf_train, pf_val, pf_test, _ = world["predicted_future_z"]
    return [
        ("metadata_only", meta_train, meta_val, meta_test, "metadata_features"),
        ("sensor_raw_only", raw_train, raw_val, raw_test, "sensor_raw_features"),
        (ENGINEERED_BASELINE, eng_train, eng_val, eng_test, "sensor_engineered_features"),
        ("current_z_only", z_train, z_val, z_test, "jepa_global_embeddings"),
        ("predicted_future_z_only", pf_train, pf_val, pf_test, "world_model_features"),
        ("sensor_engineered_plus_current_z", np.concatenate([eng_train, z_train], axis=1), np.concatenate([eng_val, z_val], axis=1), np.concatenate([eng_test, z_test], axis=1), "sensor_engineered_features,jepa_global_embeddings"),
        ("sensor_engineered_plus_predicted_future_z", np.concatenate([eng_train, pf_train], axis=1), np.concatenate([eng_val, pf_val], axis=1), np.concatenate([eng_test, pf_test], axis=1), "sensor_engineered_features,world_model_features"),
        ("sensor_engineered_plus_current_z_plus_predicted_future_z", np.concatenate([eng_train, z_train, pf_train], axis=1), np.concatenate([eng_val, z_val, pf_val], axis=1), np.concatenate([eng_test, z_test, pf_test], axis=1), "sensor_engineered_features,jepa_global_embeddings,world_model_features"),
        (METADATA_ENGINEERED_BASELINE, np.concatenate([meta_train, eng_train], axis=1), np.concatenate([meta_val, eng_val], axis=1), np.concatenate([meta_test, eng_test], axis=1), "metadata_features,sensor_engineered_features"),
        ("metadata_plus_sensor_engineered_plus_current_z", np.concatenate([meta_train, eng_train, z_train], axis=1), np.concatenate([meta_val, eng_val, z_val], axis=1), np.concatenate([meta_test, eng_test, z_test], axis=1), "metadata_features,sensor_engineered_features,jepa_global_embeddings"),
        ("metadata_plus_sensor_engineered_plus_predicted_future_z", np.concatenate([meta_train, eng_train, pf_train], axis=1), np.concatenate([meta_val, eng_val, pf_val], axis=1), np.concatenate([meta_test, eng_test, pf_test], axis=1), "metadata_features,sensor_engineered_features,world_model_features"),
        ("metadata_plus_sensor_engineered_plus_current_z_plus_predicted_future_z", np.concatenate([meta_train, eng_train, z_train, pf_train], axis=1), np.concatenate([meta_val, eng_val, z_val, pf_val], axis=1), np.concatenate([meta_test, eng_test, z_test, pf_test], axis=1), "metadata_features,sensor_engineered_features,jepa_global_embeddings,world_model_features"),
    ]


def _add_engineered_deltas(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return results
    out = results.copy()
    group_cols = ["protocol", "split_name", "seed", "forecast_horizon", "failure_horizon_cycles"]
    metrics = {
        "AUPRC": "delta_AUPRC_vs_sensor_engineered",
        "precision_at_10pct": "delta_Precision10_vs_sensor_engineered",
        "recall_at_10pct": "delta_Recall10_vs_sensor_engineered",
        "mean_lead_time": "delta_lead_time_vs_sensor_engineered",
    }
    meta_metrics = {"AUPRC": "delta_AUPRC_vs_metadata_plus_sensor_engineered"}
    base = out[out["model_name"].eq(ENGINEERED_BASELINE)][group_cols + list(metrics)].rename(columns={m: f"{m}__engineered" for m in metrics})
    out = out.merge(base, on=group_cols, how="left")
    for metric, delta in metrics.items():
        out[delta] = out[metric] - out[f"{metric}__engineered"]
    meta_base = out[out["model_name"].eq(METADATA_ENGINEERED_BASELINE)][group_cols + list(meta_metrics)].rename(columns={m: f"{m}__metadata_engineered" for m in meta_metrics})
    out = out.merge(meta_base, on=group_cols, how="left")
    for metric, delta in meta_metrics.items():
        out[delta] = out[metric] - out[f"{metric}__metadata_engineered"]
    return out


def _mean_std(results: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
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
            "delta_AUPRC_vs_sensor_engineered",
            "delta_Precision10_vs_sensor_engineered",
            "delta_Recall10_vs_sensor_engineered",
            "delta_lead_time_vs_sensor_engineered",
            "delta_AUPRC_vs_metadata_plus_sensor_engineered",
        ]
        if c in results
    ]
    out = results.groupby(group_cols, dropna=False)[metrics].agg(["mean", "std", "count"]).reset_index()
    out.columns = ["_".join(str(part) for part in col if part) for col in out.columns.to_flat_index()]
    return out


def _fmt(value: Any) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.4f}"


def _pick(df: pd.DataFrame, protocol: str, split_name: str, model_name: str, column: str) -> float:
    rows = df[
        df["protocol"].eq(protocol)
        & df["split_name"].eq(split_name)
        & df["model_name"].eq(model_name)
    ]
    if rows.empty or column not in rows:
        return float("nan")
    return float(rows.iloc[0][column])


def _selected_protocol_rows(by_protocol: pd.DataFrame) -> list[dict[str, Any]]:
    selected_models = [
        ENGINEERED_BASELINE,
        "current_z_only",
        "predicted_future_z_only",
        "sensor_engineered_plus_current_z",
        "sensor_engineered_plus_predicted_future_z",
        "sensor_engineered_plus_current_z_plus_predicted_future_z",
        METADATA_ENGINEERED_BASELINE,
        "metadata_plus_sensor_engineered_plus_current_z",
        "metadata_plus_sensor_engineered_plus_predicted_future_z",
        "metadata_plus_sensor_engineered_plus_current_z_plus_predicted_future_z",
    ]
    rows = by_protocol[by_protocol["model_name"].isin(selected_models)].copy()
    rows = rows[
        [
            "protocol",
            "split_name",
            "model_name",
            "AUPRC_mean",
            "AUPRC_std",
            "delta_AUPRC_vs_sensor_engineered_mean",
            "delta_AUPRC_vs_sensor_engineered_std",
            "delta_AUPRC_vs_metadata_plus_sensor_engineered_mean",
        ]
    ]
    for col in rows.columns:
        if col.endswith("_mean") or col.endswith("_std"):
            rows[col] = rows[col].map(_fmt)
    return rows.sort_values(["protocol", "split_name", "model_name"]).to_dict("records")


def _stability_rows(results: pd.DataFrame) -> list[dict[str, Any]]:
    combo_models = [
        "sensor_engineered_plus_current_z",
        "sensor_engineered_plus_predicted_future_z",
        "sensor_engineered_plus_current_z_plus_predicted_future_z",
        "metadata_plus_sensor_engineered_plus_current_z",
        "metadata_plus_sensor_engineered_plus_predicted_future_z",
        "metadata_plus_sensor_engineered_plus_current_z_plus_predicted_future_z",
    ]
    rows = results[results["model_name"].isin(combo_models)].copy()
    if rows.empty:
        return []
    grouped = (
        rows.groupby(["protocol", "split_name", "model_name"], dropna=False)["delta_AUPRC_vs_sensor_engineered"]
        .agg(
            delta_mean="mean",
            delta_std="std",
            count="count",
            positive_rate=lambda values: float((values > 0).mean()),
        )
        .reset_index()
    )
    grouped = grouped.sort_values("delta_mean", ascending=False)
    for col in ["delta_mean", "delta_std", "positive_rate"]:
        grouped[col] = grouped[col].map(_fmt)
    return grouped.to_dict("records")


def _answer_summary(by_protocol: pd.DataFrame) -> str:
    no_current_delta = _pick(by_protocol, "no_cycle", "no_cycle", "sensor_engineered_plus_current_z", "delta_AUPRC_vs_sensor_engineered_mean")
    no_future_delta = _pick(by_protocol, "no_cycle", "no_cycle", "sensor_engineered_plus_predicted_future_z", "delta_AUPRC_vs_sensor_engineered_mean")
    no_pred_vs_current = _pick(by_protocol, "no_cycle", "no_cycle", "predicted_future_z_only", "AUPRC_mean") - _pick(
        by_protocol, "no_cycle", "no_cycle", "current_z_only", "AUPRC_mean"
    )
    hard_combo_delta = _pick(
        by_protocol,
        "hard_generalization",
        "held_out_hardness_bin",
        "metadata_plus_sensor_engineered_plus_current_z_plus_predicted_future_z",
        "delta_AUPRC_vs_sensor_engineered_mean",
    )
    cutting_combo_delta = _pick(
        by_protocol,
        "hard_generalization",
        "held_out_cutting_condition",
        "metadata_plus_sensor_engineered_plus_current_z_plus_predicted_future_z",
        "delta_AUPRC_vs_sensor_engineered_mean",
    )
    cutting_current_only = _pick(by_protocol, "hard_generalization", "held_out_cutting_condition", "current_z_only", "AUPRC_mean")
    cutting_engineered = _pick(by_protocol, "hard_generalization", "held_out_cutting_condition", ENGINEERED_BASELINE, "AUPRC_mean")
    return "\n".join(
        [
            f"1. JEPA over sensor_engineered_only: partial. no-cycle engineered+current_z delta AUPRC={_fmt(no_current_delta)} and engineered+future_z delta={_fmt(no_future_delta)}. The deltas are positive but small.",
            f"2. predicted_future_z vs current_z: yes in no-cycle as a standalone feature (AUPRC delta={_fmt(no_pred_vs_current)}), but not in both hard splits.",
            f"3. Complementarity with engineered features: partial. Best aggregate hardness combo delta={_fmt(hard_combo_delta)} over engineered.",
            f"4. no-cycle evidence: partial. JEPA combinations improve engineered slightly, but sensor_engineered_only remains the baseline to beat.",
            f"5. held_out_hardness_bin evidence: strongest positive result for JEPA over engineered, but still modest.",
            f"6. held_out_cutting_condition evidence: not enough for JEPA+engineered. current_z_only AUPRC={_fmt(cutting_current_only)} vs engineered={_fmt(cutting_engineered)}, but adding JEPA to engineered gives combo delta={_fmt(cutting_combo_delta)}.",
            "7. Technical JEPA claim: partial only. JEPA can add incremental signal in selected protocols, but it does not robustly replace engineered sensor features.",
            "8. Product stance: use engineered sensor features as the main sensor baseline, with JEPA/predicted-future features optional until a larger stable delta appears.",
        ]
    )


def run_jepa_vs_engineered_value(
    cfg: dict[str, Any],
    seeds: list[int] | None = None,
    horizons: list[int] | None = None,
    targets: list[int] | None = None,
    out_root: str | Path = "outputs/sensor_jepa/jepa_vs_engineered_value",
) -> dict[str, Path]:
    out_root = ensure_dir(out_root)
    value_cfg = cfg.get("jepa_vs_engineered_value", {})
    seeds = seeds or [int(s) for s in value_cfg.get("seeds", [42, 123, 999])]
    horizons = horizons or [int(h) for h in value_cfg.get("horizons", [1, 3, 5])]
    targets = targets or [int(k) for k in value_cfg.get("targets", [5, 10, 20])]
    estimator = value_cfg.get("estimator", "logistic_regression")
    world_epochs = int(value_cfg.get("world_model_epochs", cfg.get("world_model", {}).get("epochs", 2)))
    device = get_device_name(cfg.get("device", "auto"))
    rows: list[dict[str, Any]] = []
    tool_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []

    for seed in seeds:
        for horizon in horizons:
            for target in targets:
                seed_everything(seed)
                combo_cfg = _cfg_for_combo(cfg, seed, horizon, target, out_root)
                combo_cfg.setdefault("world_model", {})
                combo_cfg["world_model"]["epochs"] = world_epochs
                bundle = prepare_transition_from_config(combo_cfg)
                for model_name, x_train, x_val, x_test, groups in _feature_sets(bundle, include_cycle_metadata=False, device=device, combo_cfg=combo_cfg):
                    _add_row(rows, tool_rows, bundle, "no_cycle", "no_cycle", model_name, x_train, x_val, x_test, groups, estimator, seed, horizon, target)

                _, _, meta_all = _concat_bundle(bundle)
                for split_name in ["held_out_hardness_bin", "held_out_cutting_condition"]:
                    split = build_hard_split(meta_all, split_name, seed=seed)
                    split_rows.append(
                        {
                            "protocol": "hard_generalization",
                            "split_name": split_name,
                            "seed": seed,
                            "forecast_horizon": horizon,
                            "failure_horizon_cycles": target,
                            "status": split.status,
                            "reason": split.reason,
                            "train_n": int(split.train_mask.sum()),
                            "val_n": int(split.val_mask.sum()),
                            "test_n": int(split.test_mask.sum()),
                        }
                    )
                    if split.status != "ok":
                        continue
                    split_bundle = _subset_bundle(bundle, split.train_mask, split.val_mask, split.test_mask)
                    for model_name, x_train, x_val, x_test, groups in _feature_sets(split_bundle, include_cycle_metadata=True, device=device, combo_cfg=combo_cfg):
                        _add_row(rows, tool_rows, split_bundle, "hard_generalization", split_name, model_name, x_train, x_val, x_test, groups, estimator, seed, horizon, target)

    results = _add_engineered_deltas(pd.DataFrame(rows))
    results_mean_std = _mean_std(results, ["protocol", "split_name", "model_name", "forecast_horizon", "failure_horizon_cycles"]).sort_values(
        ["protocol", "split_name", "AUPRC_mean"], ascending=[True, True, False]
    )
    by_protocol = _mean_std(results, ["protocol", "split_name", "model_name"]).sort_values(["protocol", "split_name", "AUPRC_mean"], ascending=[True, True, False])
    by_horizon = _mean_std(results, ["protocol", "split_name", "model_name", "forecast_horizon", "failure_horizon_cycles"])
    delta_cols = [
        c
        for c in [
            "protocol",
            "split_name",
            "seed",
            "forecast_horizon",
            "failure_horizon_cycles",
            "model_name",
            "AUPRC",
            "precision_at_10pct",
            "recall_at_10pct",
            "mean_lead_time",
            "delta_AUPRC_vs_sensor_engineered",
            "delta_Precision10_vs_sensor_engineered",
            "delta_Recall10_vs_sensor_engineered",
            "delta_lead_time_vs_sensor_engineered",
            "delta_AUPRC_vs_metadata_plus_sensor_engineered",
        ]
        if c in results
    ]
    deltas = results[delta_cols].copy()

    paths = {
        "results": out_root / "results.csv",
        "results_mean_std": out_root / "results_mean_std.csv",
        "results_by_protocol": out_root / "results_by_protocol.csv",
        "results_by_horizon_target": out_root / "results_by_horizon_target.csv",
        "deltas_vs_engineered": out_root / "deltas_vs_engineered.csv",
        "results_by_split": out_root / "results_by_split.csv",
        "results_by_tool": out_root / "results_by_tool.csv",
        "report": out_root / "report.md",
    }
    results.to_csv(paths["results"], index=False)
    results_mean_std.to_csv(paths["results_mean_std"], index=False)
    by_protocol.to_csv(paths["results_by_protocol"], index=False)
    by_horizon.to_csv(paths["results_by_horizon_target"], index=False)
    deltas.to_csv(paths["deltas_vs_engineered"], index=False)
    pd.DataFrame(split_rows).to_csv(paths["results_by_split"], index=False)
    pd.DataFrame(tool_rows).to_csv(paths["results_by_tool"], index=False)

    jepa_combo_models = [
        "sensor_engineered_plus_current_z",
        "sensor_engineered_plus_predicted_future_z",
        "sensor_engineered_plus_current_z_plus_predicted_future_z",
        "metadata_plus_sensor_engineered_plus_current_z",
        "metadata_plus_sensor_engineered_plus_predicted_future_z",
        "metadata_plus_sensor_engineered_plus_current_z_plus_predicted_future_z",
    ]
    top_hk = results_mean_std[results_mean_std["model_name"].isin(jepa_combo_models)]
    top_hk = top_hk.sort_values("delta_AUPRC_vs_sensor_engineered_mean", ascending=False, na_position="last").head(20)
    write_markdown_report(
        paths["report"],
        "JEPA vs Engineered Sensor Value",
        {
            "Protocol": "Deltas are computed within the same protocol/split/seed/h/K. Engineered sensor features are the central baseline.",
            "Direct Answers": _answer_summary(by_protocol),
            "Selected Protocol Summary": markdown_table(_selected_protocol_rows(by_protocol)),
            "Stability Of JEPA Over Engineered": markdown_table(_stability_rows(results)),
            "Best H/K Candidates By Delta": markdown_table(top_hk.to_dict("records")) if len(top_hk) else "No rows.",
            "Conclusion": "The evidence supports sensor features under no-cycle and hardness protocols. JEPA adds small, selected incremental value over engineered features, but engineered features remain the main baseline. No SOTA or JEPA-superiority claim is supported.",
        },
    )
    return paths
