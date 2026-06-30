from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from common.config import get_device_name
from common.forecast_metrics import forecast_metrics, threshold_from_validation
from common.paths import ensure_dir
from common.reports import markdown_table, write_markdown_report
from common.seed import seed_everything
from sensor_jepa.data.cnc_world_model import TransitionBundle, prepare_transition_from_config
from sensor_jepa.data.hard_splits import build_all_hard_splits, hard_split_report_rows
from sensor_jepa.eval.incremental_metrics import add_delta_vs_metadata
from sensor_jepa.train.hard_generalization import _concat_actions, _concat_bundle, _concat_next, _subset_bundle
from sensor_jepa.train.incremental_value_benchmark import (
    _fit_scores,
    _flatten,
    _metadata_matrix,
    engineered_sensor_features,
)
from sensor_jepa.train.no_cycle_evidence import sensor_vs_jepa_value
from sensor_jepa.train.sota_benchmark import _encode_current, _predict_future, _train_world_model


def _cfg_for_seed(cfg: dict[str, Any], seed: int, out_root: Path) -> dict[str, Any]:
    out = deepcopy(cfg)
    out["seed"] = seed
    out.setdefault("outputs", {})
    out["outputs"]["root"] = str(out_root / f"seed_{seed}")
    return out


def _evaluate_set(
    rows: list[dict[str, Any]],
    split_name: str,
    split_bundle: TransitionBundle,
    model_name: str,
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    groups: str,
    estimator: str,
    seed: int,
    horizon: int,
    target: int,
) -> None:
    val_scores, test_scores, _, train_sec, notes = _fit_scores(x_train, split_bundle.y_failure_train, x_val, x_test, estimator, seed)
    threshold = threshold_from_validation(split_bundle.y_failure_val, val_scores)
    row = forecast_metrics(
        split_bundle.y_failure_test,
        test_scores,
        threshold=threshold,
        tool_ids=split_bundle.test_meta["ToolIndex"].to_numpy() if "ToolIndex" in split_bundle.test_meta else None,
        cycle_to_failure=split_bundle.test_meta["CycleToFailure"].to_numpy() if "CycleToFailure" in split_bundle.test_meta else None,
    )
    row.update(
        {
            "split_name": split_name,
            "status": "ok",
            "model_name": model_name,
            "model_family": "hard_generalization_evidence",
            "feature_groups": groups,
            "estimator": estimator,
            "protocol": "hard_generalization",
            "seed": seed,
            "forecast_horizon": horizon,
            "failure_horizon_cycles": target,
            "train_time_sec": train_sec,
            "notes": notes,
        }
    )
    rows.append(row)


def _mean_std(results: pd.DataFrame) -> pd.DataFrame:
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
            "delta_AUPRC_vs_metadata_only",
            "delta_AUROC_vs_metadata_only",
            "delta_Precision@10_vs_metadata_only",
            "delta_Recall@10_vs_metadata_only",
        ]
        if c in results
    ]
    groups = ["split_name", "model_name", "forecast_horizon", "failure_horizon_cycles"]
    out = results.groupby(groups, dropna=False)[metrics].agg(["mean", "std", "count"]).reset_index()
    out.columns = ["_".join(str(part) for part in col if part) for col in out.columns.to_flat_index()]
    return out.sort_values(["split_name", "AUPRC_mean"], ascending=[True, False]).reset_index(drop=True)


def run_hard_generalization_evidence(
    cfg: dict[str, Any],
    seeds: list[int] | None = None,
    out_root: str | Path = "outputs/sensor_jepa/hard_generalization_evidence",
) -> dict[str, Path]:
    out_root = ensure_dir(out_root)
    evidence_cfg = cfg.get("hard_generalization_evidence", {})
    seeds = seeds or [int(s) for s in evidence_cfg.get("seeds", [42, 123, 999])]
    estimator = evidence_cfg.get("estimator", "logistic_regression")
    world_epochs = int(evidence_cfg.get("world_model_epochs", cfg.get("world_model", {}).get("epochs", 2)))
    split_filter = set(evidence_cfg.get("splits", ["held_out_tool_id", "held_out_hardness_bin", "held_out_cutting_condition"]))
    horizon = int(cfg.get("world_model", {}).get("forecast_horizon", 3))
    target = int(cfg.get("world_model", {}).get("failure_horizon_cycles", 10))
    device = get_device_name(cfg.get("device", "auto"))
    all_rows: list[dict[str, Any]] = []
    all_split_rows: list[dict[str, Any]] = []

    for seed in seeds:
        seed_everything(seed)
        combo_cfg = _cfg_for_seed(cfg, seed, out_root)
        combo_cfg.setdefault("world_model", {})
        combo_cfg["world_model"]["epochs"] = world_epochs
        bundle = prepare_transition_from_config(combo_cfg)
        _, _, meta_all = _concat_bundle(bundle)
        splits = [s for s in build_all_hard_splits(meta_all, seed=seed) if s.name in split_filter]
        split_rows = hard_split_report_rows(splits)
        for row in split_rows:
            row["seed"] = seed
        all_split_rows.extend(split_rows)

        for split in splits:
            if split.status != "ok":
                all_rows.append({"split_name": split.name, "status": split.status, "model_name": "pending", "seed": seed, "reason": split.reason})
                continue
            split_bundle = _subset_bundle(bundle, split.train_mask, split.val_mask, split.test_mask)
            meta_train, _ = _metadata_matrix(split_bundle, "train", include_cycle=True)
            meta_val, _ = _metadata_matrix(split_bundle, "val", include_cycle=True)
            meta_test, _ = _metadata_matrix(split_bundle, "test", include_cycle=True)
            raw_train = _flatten(split_bundle.x_train)
            raw_val = _flatten(split_bundle.x_val)
            raw_test = _flatten(split_bundle.x_test)
            eng_train, _ = engineered_sensor_features(split_bundle.x_train)
            eng_val, _ = engineered_sensor_features(split_bundle.x_val)
            eng_test, _ = engineered_sensor_features(split_bundle.x_test)

            hard_cfg = deepcopy(combo_cfg)
            model, _ = _train_world_model(hard_cfg, split_bundle, device, use_actions=True, pretrained_encoder=False)
            z_train = _encode_current(model, split_bundle.x_train, device)
            z_val = _encode_current(model, split_bundle.x_val, device)
            z_test = _encode_current(model, split_bundle.x_test, device)
            pf_train = _predict_future(model, split_bundle.x_train, split_bundle.a_train, device)
            pf_val = _predict_future(model, split_bundle.x_val, split_bundle.a_val, device)
            pf_test = _predict_future(model, split_bundle.x_test, split_bundle.a_test, device)

            feature_sets = [
                ("metadata_only", meta_train, meta_val, meta_test, "metadata_features,cycle_features"),
                ("sensor_raw_only", raw_train, raw_val, raw_test, "sensor_raw_features"),
                ("sensor_engineered_only", eng_train, eng_val, eng_test, "sensor_engineered_features"),
                ("current_z_only", z_train, z_val, z_test, "jepa_global_embeddings"),
                ("predicted_future_z_only", pf_train, pf_val, pf_test, "world_model_features"),
                ("metadata_plus_current_z", np.concatenate([meta_train, z_train], axis=1), np.concatenate([meta_val, z_val], axis=1), np.concatenate([meta_test, z_test], axis=1), "metadata_features,cycle_features,jepa_global_embeddings"),
                ("metadata_plus_predicted_future_z", np.concatenate([meta_train, pf_train], axis=1), np.concatenate([meta_val, pf_val], axis=1), np.concatenate([meta_test, pf_test], axis=1), "metadata_features,cycle_features,world_model_features"),
                ("metadata_plus_current_z_plus_predicted_future_z", np.concatenate([meta_train, z_train, pf_train], axis=1), np.concatenate([meta_val, z_val, pf_val], axis=1), np.concatenate([meta_test, z_test, pf_test], axis=1), "metadata_features,cycle_features,jepa_global_embeddings,world_model_features"),
                ("sensor_raw_plus_current_z", np.concatenate([raw_train, z_train], axis=1), np.concatenate([raw_val, z_val], axis=1), np.concatenate([raw_test, z_test], axis=1), "sensor_raw_features,jepa_global_embeddings"),
                ("metadata_plus_sensor_raw", np.concatenate([meta_train, raw_train], axis=1), np.concatenate([meta_val, raw_val], axis=1), np.concatenate([meta_test, raw_test], axis=1), "metadata_features,cycle_features,sensor_raw_features"),
                ("metadata_plus_sensor_raw_plus_current_z", np.concatenate([meta_train, raw_train, z_train], axis=1), np.concatenate([meta_val, raw_val, z_val], axis=1), np.concatenate([meta_test, raw_test, z_test], axis=1), "metadata_features,cycle_features,sensor_raw_features,jepa_global_embeddings"),
            ]
            for model_name, x_train, x_val, x_test, groups in feature_sets:
                _evaluate_set(all_rows, split.name, split_bundle, model_name, x_train, x_val, x_test, groups, estimator, seed, horizon, target)

    results = pd.DataFrame(all_rows)
    ok = results[results["status"].eq("ok")].copy() if "status" in results else pd.DataFrame()
    if not ok.empty:
        ok = add_delta_vs_metadata(
            ok,
            baseline_model="metadata_only",
            group_columns=["protocol", "split_name", "seed", "forecast_horizon", "failure_horizon_cycles"],
        )
        pending = results[~results["status"].eq("ok")].copy()
        results = pd.concat([ok, pending], ignore_index=True, sort=False)
    mean_std = _mean_std(ok)
    value_detail, value_summary = sensor_vs_jepa_value(ok.rename(columns={"split_name": "split_name"})) if not ok.empty else (pd.DataFrame(), pd.DataFrame())
    paths = {
        "results": out_root / "results.csv",
        "results_mean_std": out_root / "results_mean_std.csv",
        "results_by_split": out_root / "results_by_split.csv",
        "report": out_root / "report.md",
        "sensor_vs_jepa_detail": Path("outputs/sensor_jepa/sensor_vs_jepa_value/hard_generalization_detail.csv"),
        "sensor_vs_jepa_summary": Path("outputs/sensor_jepa/sensor_vs_jepa_value/hard_generalization_summary.csv"),
    }
    results.to_csv(paths["results"], index=False)
    mean_std.to_csv(paths["results_mean_std"], index=False)
    pd.DataFrame(all_split_rows).to_csv(paths["results_by_split"], index=False)
    ensure_dir(paths["sensor_vs_jepa_detail"].parent)
    value_detail.to_csv(paths["sensor_vs_jepa_detail"], index=False)
    value_summary.to_csv(paths["sensor_vs_jepa_summary"], index=False)

    top = mean_std.head(30)
    write_markdown_report(
        paths["report"],
        "Hard Generalization Sensor Evidence",
        {
            "Protocol": "Held-out metadata groups. Deltas are computed against metadata-only within same split and seed.",
            "Split Availability": markdown_table(pd.DataFrame(all_split_rows).to_dict("records")),
            "Top Mean/Std Rows": markdown_table(top.to_dict("records")) if len(top) else "No completed rows.",
            "Raw vs JEPA Deltas": markdown_table(value_summary.head(30).to_dict("records")) if len(value_summary) else "No comparison rows.",
            "Conclusion Rule": "Use split-level deltas. A model can help in one hard split and fail in another.",
        },
    )
    return paths
