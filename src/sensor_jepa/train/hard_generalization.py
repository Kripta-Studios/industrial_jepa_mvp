from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from common.config import get_device_name
from common.forecast_metrics import forecast_metrics, threshold_from_validation
from common.paths import ensure_dir
from common.reports import markdown_table, write_markdown_report
from sensor_jepa.data.cnc_world_model import TransitionBundle, prepare_transition_from_config
from sensor_jepa.data.hard_splits import build_all_hard_splits, hard_split_report_rows
from sensor_jepa.eval.incremental_metrics import add_delta_vs_metadata
from sensor_jepa.train.incremental_value_benchmark import _cycle_matrix, _fit_scores, _flatten, _metadata_matrix
from sensor_jepa.train.sota_benchmark import _encode_current, _predict_future, _train_world_model


def _concat_bundle(bundle: TransitionBundle) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    x = np.concatenate([bundle.x_train, bundle.x_val, bundle.x_test], axis=0)
    y = np.concatenate([bundle.y_failure_train, bundle.y_failure_val, bundle.y_failure_test], axis=0)
    meta = pd.concat([bundle.train_meta, bundle.val_meta, bundle.test_meta], ignore_index=True)
    return x, y, meta


def _concat_actions(bundle: TransitionBundle) -> np.ndarray:
    return np.concatenate([bundle.a_train, bundle.a_val, bundle.a_test], axis=0)


def _concat_next(bundle: TransitionBundle) -> np.ndarray:
    return np.concatenate([bundle.x_next_train, bundle.x_next_val, bundle.x_next_test], axis=0)


def _subset_bundle(bundle: TransitionBundle, train_mask: np.ndarray, val_mask: np.ndarray, test_mask: np.ndarray) -> TransitionBundle:
    x_all, y_all, meta_all = _concat_bundle(bundle)
    a_all = _concat_actions(bundle)
    x_next_all = _concat_next(bundle)
    return TransitionBundle(
        x_train=x_all[train_mask],
        a_train=a_all[train_mask],
        x_next_train=x_next_all[train_mask],
        y_failure_train=y_all[train_mask],
        x_val=x_all[val_mask],
        a_val=a_all[val_mask],
        x_next_val=x_next_all[val_mask],
        y_failure_val=y_all[val_mask],
        x_test=x_all[test_mask],
        a_test=a_all[test_mask],
        x_next_test=x_next_all[test_mask],
        y_failure_test=y_all[test_mask],
        train_meta=meta_all.loc[train_mask].reset_index(drop=True),
        val_meta=meta_all.loc[val_mask].reset_index(drop=True),
        test_meta=meta_all.loc[test_mask].reset_index(drop=True),
        feature_names=bundle.feature_names,
        action_names=bundle.action_names,
        standardizer=bundle.standardizer,
        action_standardizer=bundle.action_standardizer,
    )


def _select(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return arr[mask]


def run_hard_generalization(cfg: dict[str, Any]) -> dict[str, Path]:
    out_dir = ensure_dir(cfg.get("hard_generalization", {}).get("output_dir", "outputs/sensor_jepa/hard_generalization"))
    seed = int(cfg.get("seed", 42))
    estimator = cfg.get("hard_generalization", {}).get("estimator", "logistic_regression")
    include_latents = bool(cfg.get("hard_generalization", {}).get("include_latents", True))
    device = get_device_name(cfg.get("device", "auto"))
    bundle = prepare_transition_from_config(cfg)
    x_all, y_all, meta_all = _concat_bundle(bundle)
    splits = build_all_hard_splits(meta_all, seed=seed)
    split_rows = hard_split_report_rows(splits)
    rows: list[dict[str, Any]] = []

    for split in splits:
        if split.status != "ok":
            rows.append(
                {
                    "split_name": split.name,
                    "status": split.status,
                    "model_name": "pending",
                    "reason": split.reason,
                }
            )
            continue
        split_bundle = _subset_bundle(bundle, split.train_mask, split.val_mask, split.test_mask)
        train_meta, meta_names = _metadata_matrix(split_bundle, "train", include_cycle=True)
        val_meta, _ = _metadata_matrix(split_bundle, "val", include_cycle=True)
        test_meta, _ = _metadata_matrix(split_bundle, "test", include_cycle=True)
        train_cycle, cycle_names = _cycle_matrix(split_bundle, "train")
        val_cycle, _ = _cycle_matrix(split_bundle, "val")
        test_cycle, _ = _cycle_matrix(split_bundle, "test")
        feature_sets = [
            ("metadata_only", train_meta, val_meta, test_meta, meta_names, "metadata_features,cycle_features"),
            ("cycle_only", train_cycle, val_cycle, test_cycle, cycle_names, "cycle_features"),
            ("sensor_raw_only", split_bundle.x_train, split_bundle.x_val, split_bundle.x_test, split_bundle.feature_names, "sensor_raw_features"),
        ]
        if include_latents:
            hard_cfg = dict(cfg)
            hard_cfg.setdefault("world_model", {})
            hard_cfg["world_model"]["epochs"] = int(cfg.get("hard_generalization", {}).get("world_model_epochs", cfg.get("world_model", {}).get("epochs", 2)))
            model, _ = _train_world_model(hard_cfg, split_bundle, device, use_actions=True, pretrained_encoder=False)
            z_train = _encode_current(model, split_bundle.x_train, device)
            z_val = _encode_current(model, split_bundle.x_val, device)
            z_test = _encode_current(model, split_bundle.x_test, device)
            pf_train = _predict_future(model, split_bundle.x_train, split_bundle.a_train, device)
            pf_val = _predict_future(model, split_bundle.x_val, split_bundle.a_val, device)
            pf_test = _predict_future(model, split_bundle.x_test, split_bundle.a_test, device)
            z_names = [f"current_z_{i}" for i in range(z_train.shape[1])]
            pf_names = [f"predicted_future_z_{i}" for i in range(pf_train.shape[1])]
            feature_sets.extend(
                [
                    ("current_z_only", z_train, z_val, z_test, z_names, "jepa_global_embeddings"),
                    ("predicted_future_z_only", pf_train, pf_val, pf_test, pf_names, "world_model_features"),
                    (
                        "metadata_plus_current_z",
                        np.concatenate([train_meta, z_train], axis=1),
                        np.concatenate([val_meta, z_val], axis=1),
                        np.concatenate([test_meta, z_test], axis=1),
                        meta_names + z_names,
                        "metadata_features,cycle_features,jepa_global_embeddings",
                    ),
                    (
                        "metadata_plus_current_z_plus_predicted_future_z",
                        np.concatenate([train_meta, z_train, pf_train], axis=1),
                        np.concatenate([val_meta, z_val, pf_val], axis=1),
                        np.concatenate([test_meta, z_test, pf_test], axis=1),
                        meta_names + z_names + pf_names,
                        "metadata_features,cycle_features,jepa_global_embeddings,world_model_features",
                    ),
                ]
            )
        for model_name, xtr, xva, xte, feature_names, groups in feature_sets:
            ytr = split_bundle.y_failure_train
            yva = split_bundle.y_failure_val
            yte = split_bundle.y_failure_test
            val_scores, test_scores, _, train_sec, notes = _fit_scores(xtr, ytr, xva, xte, estimator, seed)
            threshold = threshold_from_validation(yva, val_scores)
            metrics = forecast_metrics(
                yte,
                test_scores,
                threshold=threshold,
                tool_ids=split_bundle.test_meta["ToolIndex"].to_numpy() if "ToolIndex" in split_bundle.test_meta else None,
                cycle_to_failure=split_bundle.test_meta["CycleToFailure"].to_numpy() if "CycleToFailure" in split_bundle.test_meta else None,
            )
            metrics.update(
                {
                    "split_name": split.name,
                    "status": "ok",
                    "model_name": model_name,
                    "model_family": "hard_generalization",
                    "feature_groups": groups,
                    "estimator": estimator,
                    "protocol": "hard_generalization",
                    "seed": seed,
                    "forecast_horizon": int(cfg.get("world_model", {}).get("forecast_horizon", 1)),
                    "failure_horizon_cycles": int(cfg.get("world_model", {}).get("failure_horizon_cycles", 10)),
                    "train_time_sec": train_sec,
                    "notes": notes,
                    "held_out_groups": ",".join(split.test_groups),
                }
            )
            rows.append(metrics)

    results = add_delta_vs_metadata(
        pd.DataFrame(rows),
        baseline_model="metadata_only",
        group_columns=["protocol", "split_name", "seed", "forecast_horizon", "failure_horizon_cycles"],
    )
    results_path = out_dir / "results.csv"
    split_path = out_dir / "results_by_split.csv"
    report_path = out_dir / "report.md"
    results.to_csv(results_path, index=False)
    pd.DataFrame(split_rows).to_csv(split_path, index=False)
    ok = results[results.get("status", "").eq("ok")] if "status" in results else pd.DataFrame()
    write_markdown_report(
        report_path,
        "Sensor Hard Generalization Report",
        {
            "Status": "Hard splits are reported as ok or pending per available CNC metadata column.",
            "Interpretation Rule": "Sensor value is judged by delta over metadata-only inside each held-out split.",
            "Top Rows": markdown_table(ok.sort_values("AUPRC", ascending=False).head(20).to_dict("records")) if len(ok) else "No completed hard split rows.",
            "Split Availability": markdown_table(split_rows),
        },
    )
    return {"results": results_path, "results_by_split": split_path, "report": report_path}
