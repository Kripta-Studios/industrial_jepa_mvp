from __future__ import annotations

import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common.config import get_device_name
from common.forecast_metrics import forecast_metrics, metrics_by_tool, threshold_from_validation
from common.paths import ensure_dir
from common.reports import markdown_table, write_json, write_markdown_report
from common.seed import seed_everything
from sensor_jepa.data.cnc_world_model import TransitionBundle, prepare_transition_from_config
from sensor_jepa.eval.incremental_metrics import add_delta_vs_metadata, feature_group_ablation, summarize_by_horizon_target, summarize_by_protocol
from sensor_jepa.models.official_time_series_baselines import availability_as_dict, baseline_metadata_for_name
from sensor_jepa.models.strong_baselines import predict_scores
from sensor_jepa.train.sota_benchmark import _encode_current, _predict_future, _train_world_model, feature_audit, leakage_report


CORE_METRICS = ["AUROC", "AUPRC", "precision_at_10pct", "recall_at_10pct", "false_alarms_per_tool", "mean_lead_time"]


def _cfg_for_combo(cfg: dict[str, Any], seed: int, horizon: int, target: int, out_root: Path) -> dict[str, Any]:
    combo = deepcopy(cfg)
    combo["seed"] = seed
    combo.setdefault("world_model", {})
    combo["world_model"]["forecast_horizon"] = horizon
    combo["world_model"]["failure_horizon_cycles"] = target
    combo.setdefault("outputs", {})
    combo["outputs"]["root"] = str(out_root / f"seed_{seed}" / f"h_{horizon}" / f"k_{target}")
    return combo


def _meta_col(bundle: TransitionBundle, split: str, col: str) -> np.ndarray:
    meta = getattr(bundle, f"{split}_meta")
    if len(meta) and col in meta:
        return pd.to_numeric(meta[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    return np.zeros(len(getattr(bundle, f"y_failure_{split}")), dtype=np.float32)


def _metadata_matrix(bundle: TransitionBundle, split: str, include_cycle: bool) -> tuple[np.ndarray, list[str]]:
    meta = getattr(bundle, f"{split}_meta")
    names = [c for c in bundle.action_names if c in meta.columns]
    values = [_meta_col(bundle, split, c) for c in names]
    if include_cycle:
        for col in ["source_cycle", "NumberOfCycle"]:
            if col in meta.columns:
                names.append(col)
                values.append(_meta_col(bundle, split, col))
    if not values:
        return np.zeros((len(getattr(bundle, f"y_failure_{split}")), 1), dtype=np.float32), ["constant"]
    return np.stack(values, axis=1).astype(np.float32), names


def _cycle_matrix(bundle: TransitionBundle, split: str) -> tuple[np.ndarray, list[str]]:
    names = [c for c in ["source_cycle", "NumberOfCycle"] if c in getattr(bundle, f"{split}_meta").columns]
    if not names:
        return np.zeros((len(getattr(bundle, f"y_failure_{split}")), 1), dtype=np.float32), ["constant_cycle"]
    return np.stack([_meta_col(bundle, split, c) for c in names], axis=1), names


def _flatten(x: np.ndarray) -> np.ndarray:
    return np.nan_to_num(x.reshape(len(x), -1), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def engineered_sensor_features(x: np.ndarray) -> tuple[np.ndarray, list[str]]:
    x = np.nan_to_num(x.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    eps = 1e-6
    mean = x.mean(axis=1)
    std = x.std(axis=1)
    rms = np.sqrt(np.mean(x * x, axis=1))
    minv = x.min(axis=1)
    maxv = x.max(axis=1)
    ptp = maxv - minv
    centered = x - mean[:, None, :]
    skew = np.mean(centered**3, axis=1) / np.maximum(std**3, eps)
    kurt = np.mean(centered**4, axis=1) / np.maximum(std**4, eps)
    crest = np.max(np.abs(x), axis=1) / np.maximum(rms, eps)
    energy = np.sum(x * x, axis=1)
    zcr = np.mean(np.diff(np.signbit(x), axis=1), axis=1).astype(np.float32)
    fft = np.abs(np.fft.rfft(x, axis=1))
    freqs = np.fft.rfftfreq(x.shape[1], d=1.0).astype(np.float32)
    power = fft**2
    total_power = np.maximum(power.sum(axis=1), eps)
    centroid = (power * freqs[None, :, None]).sum(axis=1) / total_power
    bandwidth = np.sqrt(((freqs[None, :, None] - centroid[:, None, :]) ** 2 * power).sum(axis=1) / total_power)
    dominant = freqs[np.argmax(power, axis=1)]
    low = power[:, : max(1, power.shape[1] // 3), :].sum(axis=1)
    mid = power[:, max(1, power.shape[1] // 3) : max(2, 2 * power.shape[1] // 3), :].sum(axis=1)
    high = power[:, max(2, 2 * power.shape[1] // 3) :, :].sum(axis=1)
    t = np.arange(x.shape[1], dtype=np.float32)
    t = (t - t.mean()) / max(float(t.std()), eps)
    trend = (x * t[None, :, None]).mean(axis=1)
    parts = [mean, std, rms, ptp, minv, maxv, skew, kurt, crest, energy, zcr, centroid, bandwidth, dominant, low, mid, high, trend]
    names = [
        "mean",
        "std",
        "rms",
        "peak_to_peak",
        "min",
        "max",
        "skewness",
        "kurtosis",
        "crest_factor",
        "energy",
        "zero_crossing_rate",
        "spectral_centroid",
        "spectral_bandwidth",
        "dominant_frequency",
        "band_power_low",
        "band_power_mid",
        "band_power_high",
        "rolling_trend",
    ]
    return np.concatenate(parts, axis=1).astype(np.float32), names


def _make_estimator(name: str, seed: int):
    if name == "logistic_regression":
        return make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed))
    if name == "hist_gradient_boosting" or name == "gbt":
        return HistGradientBoostingClassifier(random_state=seed, max_iter=200, learning_rate=0.05)
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=-1)
    if name == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=seed,
            n_jobs=2,
        )
    if name == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=15,
            class_weight="balanced",
            random_state=seed,
            verbosity=-1,
        )
    raise ValueError(f"Unknown estimator: {name}")


def _fit_scores(x_train, y_train, x_val, x_test, estimator: str, seed: int):
    x_train = _flatten(x_train)
    x_val = _flatten(x_val)
    x_test = _flatten(x_test)
    start = time.time()
    if len(np.unique(y_train)) < 2:
        constant = float(np.mean(y_train)) if len(y_train) else 0.0
        return np.full(len(x_val), constant), np.full(len(x_test), constant), None, time.time() - start, "single_class_train"
    model = _make_estimator(estimator, seed)
    model.fit(x_train, y_train)
    return predict_scores(model, x_val), predict_scores(model, x_test), model, time.time() - start, ""


def _row_from_scores(
    bundle: TransitionBundle,
    model_name: str,
    model_family: str,
    protocol: str,
    feature_groups: str,
    estimator: str,
    val_scores: np.ndarray,
    test_scores: np.ndarray,
    seed: int,
    horizon: int,
    target: int,
    train_time_sec: float,
    notes: str = "",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    threshold = threshold_from_validation(bundle.y_failure_val, val_scores)
    meta = bundle.test_meta
    tool_ids = meta["ToolIndex"].to_numpy() if len(meta) and "ToolIndex" in meta else None
    ctf = meta["CycleToFailure"].to_numpy() if len(meta) and "CycleToFailure" in meta else None
    row = forecast_metrics(bundle.y_failure_test, test_scores, threshold=threshold, tool_ids=tool_ids, cycle_to_failure=ctf)
    row.update(
        {
            "dataset": "cnc_milling",
            "task": "failure_soon_prediction",
            "protocol": protocol,
            "model_name": model_name,
            "model_family": model_family,
            "estimator": estimator,
            "feature_groups": feature_groups,
            "seed": seed,
            "forecast_horizon": horizon,
            "failure_horizon_cycles": target,
            "threshold_source": "validation_best_f1",
            "train_time_sec": train_time_sec,
            "test_failure_rate": float(np.mean(bundle.y_failure_test)) if len(bundle.y_failure_test) else 0.0,
            "notes": notes,
        }
    )
    row.update(baseline_metadata_for_name(model_name, model_family, notes))
    tool_rows = metrics_by_tool(bundle.y_failure_test, test_scores, threshold, tool_ids if tool_ids is not None else np.zeros(len(test_scores)))
    for tool_row in tool_rows:
        tool_row.update({"model_name": model_name, "protocol": protocol, "seed": seed, "forecast_horizon": horizon, "failure_horizon_cycles": target})
    return row, tool_rows


def _world_features(cfg: dict[str, Any], bundle: TransitionBundle, device: str) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]]:
    use_actions = True
    model, _ = _train_world_model(cfg, bundle, device, use_actions=use_actions, pretrained_encoder=False)
    z_train = _encode_current(model, bundle.x_train, device)
    z_val = _encode_current(model, bundle.x_val, device)
    z_test = _encode_current(model, bundle.x_test, device)
    pf_train = _predict_future(model, bundle.x_train, bundle.a_train, device)
    pf_val = _predict_future(model, bundle.x_val, bundle.a_val, device)
    pf_test = _predict_future(model, bundle.x_test, bundle.a_test, device)

    @torch.no_grad()
    def score(x, a, x_next):
        scores = []
        for i in range(0, len(x), 256):
            xb = torch.tensor(x[i : i + 256], dtype=torch.float32, device=device)
            ab = torch.tensor(a[i : i + 256], dtype=torch.float32, device=device)
            xnb = torch.tensor(x_next[i : i + 256], dtype=torch.float32, device=device)
            out = model(xb, ab, xnb)
            scores.append(out["per_sample_error"].cpu().numpy())
        return np.concatenate(scores).reshape(-1, 1)

    score_train = score(bundle.x_train, bundle.a_train, bundle.x_next_train)
    score_val = score(bundle.x_val, bundle.a_val, bundle.x_next_val)
    score_test = score(bundle.x_test, bundle.a_test, bundle.x_next_test)
    return {
        "current_z": (z_train, z_val, z_test, [f"current_z_{i}" for i in range(z_train.shape[1])]),
        "predicted_future_z": (pf_train, pf_val, pf_test, [f"predicted_future_z_{i}" for i in range(pf_train.shape[1])]),
        "world_model_score": (score_train, score_val, score_test, ["world_model_score"]),
    }


def _available_gbt_estimators(requested: list[str]) -> list[str]:
    available = []
    for name in requested:
        try:
            _make_estimator(name, 42)
            available.append(name)
        except Exception:
            continue
    return available


def _run_feature_row(
    bundle: TransitionBundle,
    rows: list[dict[str, Any]],
    tool_rows: list[dict[str, Any]],
    importance_rows: list[dict[str, Any]],
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    feature_names: list[str],
    model_name: str,
    model_family: str,
    protocol: str,
    feature_groups: str,
    estimator: str,
    seed: int,
    horizon: int,
    target: int,
    permutation: bool = False,
) -> None:
    val_scores, test_scores, model, train_sec, notes = _fit_scores(x_train, bundle.y_failure_train, x_val, x_test, estimator, seed)
    row, by_tool = _row_from_scores(bundle, model_name, model_family, protocol, feature_groups, estimator, val_scores, test_scores, seed, horizon, target, train_sec, notes)
    rows.append(row)
    tool_rows.extend(by_tool)
    if model is not None:
        estimator_obj = model.steps[-1][1] if hasattr(model, "steps") else model
        importances = getattr(estimator_obj, "feature_importances_", None)
        if importances is not None:
            for name, value in sorted(zip(feature_names, importances), key=lambda item: float(item[1]), reverse=True)[:100]:
                importance_rows.append({"model_name": model_name, "protocol": protocol, "feature": name, "importance": float(value), "importance_type": "native"})
        if permutation and len(feature_names) <= 256:
            try:
                result = permutation_importance(model, _flatten(x_val), bundle.y_failure_val, n_repeats=3, random_state=seed, scoring="average_precision")
                for name, value in sorted(zip(feature_names, result.importances_mean), key=lambda item: float(item[1]), reverse=True)[:100]:
                    importance_rows.append({"model_name": model_name, "protocol": protocol, "feature": name, "importance": float(value), "importance_type": "permutation_auprc"})
            except Exception:
                pass


def run_incremental_value_benchmark(cfg: dict[str, Any]) -> dict[str, Path]:
    bench_cfg = cfg.get("incremental_benchmark", {})
    out_root = ensure_dir(bench_cfg.get("output_dir", "outputs/sensor_jepa/incremental_value_benchmark"))
    seeds = [int(s) for s in bench_cfg.get("seeds", [int(cfg.get("seed", 42))])]
    horizons = [int(h) for h in bench_cfg.get("horizons", [int(cfg.get("world_model", {}).get("forecast_horizon", 3))])]
    targets = [int(k) for k in bench_cfg.get("targets", [int(cfg.get("world_model", {}).get("failure_horizon_cycles", 10))])]
    protocols = bench_cfg.get("protocols", ["operational", "no_cycle", "sensor_only", "metadata_only"])
    core_estimator = bench_cfg.get("core_estimator", "logistic_regression")
    gbt_estimators = _available_gbt_estimators(bench_cfg.get("gbt_estimators", ["hist_gradient_boosting", "xgboost", "lightgbm"]))
    include_world = bool(bench_cfg.get("include_world_model", True))
    permutation = bool(bench_cfg.get("permutation_importance", False))
    device = get_device_name(cfg.get("device", "auto"))

    all_rows: list[dict[str, Any]] = []
    all_tool_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    leakage_checks = []
    first_audit_written = False

    for seed in seeds:
        for horizon in horizons:
            for target in targets:
                combo_cfg = _cfg_for_combo(cfg, seed, horizon, target, out_root)
                combo_cfg.setdefault("world_model", {})
                combo_cfg["world_model"]["epochs"] = int(bench_cfg.get("world_model_epochs", combo_cfg["world_model"].get("epochs", 4)))
                seed_everything(seed)
                bundle = prepare_transition_from_config(combo_cfg)
                leakage_checks.append({"seed": seed, "forecast_horizon": horizon, "failure_horizon_cycles": target, **leakage_report(bundle)})
                if not first_audit_written:
                    audit_df, forbidden = feature_audit(bundle)
                    audit_df.to_csv(out_root / "feature_audit.csv", index=False)
                    write_json(out_root / "forbidden_columns_report.json", forbidden)
                    first_audit_written = True

                meta_cycle = {split: _metadata_matrix(bundle, split, include_cycle=True) for split in ["train", "val", "test"]}
                meta_no_cycle = {split: _metadata_matrix(bundle, split, include_cycle=False) for split in ["train", "val", "test"]}
                cycle = {split: _cycle_matrix(bundle, split) for split in ["train", "val", "test"]}
                engineered = {
                    "train": engineered_sensor_features(bundle.x_train),
                    "val": engineered_sensor_features(bundle.x_val),
                    "test": engineered_sensor_features(bundle.x_test),
                }
                world = _world_features(combo_cfg, bundle, device) if include_world else {}

                def add_core(protocol: str, name: str, train, val, test, names, groups: str) -> None:
                    _run_feature_row(
                        bundle,
                        all_rows,
                        all_tool_rows,
                        importance_rows,
                        train,
                        val,
                        test,
                        names,
                        name,
                        "incremental_value",
                        protocol,
                        groups,
                        core_estimator,
                        seed,
                        horizon,
                        target,
                        permutation=False,
                    )

                for protocol in protocols:
                    meta = meta_cycle if protocol in {"operational", "metadata_only"} else meta_no_cycle
                    if protocol != "sensor_only":
                        add_core(protocol, "metadata_only", meta["train"][0], meta["val"][0], meta["test"][0], meta["train"][1], "metadata_features,cycle_features" if protocol in {"operational", "metadata_only"} else "metadata_features")
                        add_core(protocol, "actions_only", bundle.a_train, bundle.a_val, bundle.a_test, bundle.action_names, "action_context_features")
                    if protocol in {"operational", "metadata_only"}:
                        add_core(protocol, "cycle_only", cycle["train"][0], cycle["val"][0], cycle["test"][0], cycle["train"][1], "cycle_features")
                    if protocol != "metadata_only":
                        add_core(protocol, "sensor_raw_only", bundle.x_train, bundle.x_val, bundle.x_test, bundle.feature_names, "sensor_raw_features")
                        if "current_z" in world:
                            ztr, zva, zte, znames = world["current_z"]
                            pftr, pfva, pfte, pfnames = world["predicted_future_z"]
                            wstr, wsva, wste, wnames = world["world_model_score"]
                            add_core(protocol, "current_z_only", ztr, zva, zte, znames, "jepa_global_embeddings")
                            add_core(protocol, "predicted_future_z_only", pftr, pfva, pfte, pfnames, "world_model_features")
                            add_core(protocol, "world_model_score_only", wstr, wsva, wste, wnames, "world_model_features")
                            if protocol != "sensor_only":
                                add_core(protocol, "metadata_plus_current_z", np.concatenate([meta["train"][0], ztr], axis=1), np.concatenate([meta["val"][0], zva], axis=1), np.concatenate([meta["test"][0], zte], axis=1), meta["train"][1] + znames, "metadata_features,jepa_global_embeddings")
                                add_core(protocol, "metadata_plus_predicted_future_z", np.concatenate([meta["train"][0], pftr], axis=1), np.concatenate([meta["val"][0], pfva], axis=1), np.concatenate([meta["test"][0], pfte], axis=1), meta["train"][1] + pfnames, "metadata_features,world_model_features")
                                add_core(protocol, "metadata_plus_world_model_score", np.concatenate([meta["train"][0], wstr], axis=1), np.concatenate([meta["val"][0], wsva], axis=1), np.concatenate([meta["test"][0], wste], axis=1), meta["train"][1] + wnames, "metadata_features,world_model_features")
                                add_core(protocol, "metadata_plus_current_z_plus_world_model_score", np.concatenate([meta["train"][0], ztr, wstr], axis=1), np.concatenate([meta["val"][0], zva, wsva], axis=1), np.concatenate([meta["test"][0], zte, wste], axis=1), meta["train"][1] + znames + wnames, "metadata_features,jepa_global_embeddings,world_model_features")
                                add_core(protocol, "metadata_plus_current_z_plus_predicted_future_z", np.concatenate([meta["train"][0], ztr, pftr], axis=1), np.concatenate([meta["val"][0], zva, pfva], axis=1), np.concatenate([meta["test"][0], zte, pfte], axis=1), meta["train"][1] + znames + pfnames, "metadata_features,jepa_global_embeddings,world_model_features")
                                add_core(protocol, "metadata_plus_sensor_raw_plus_current_z", np.concatenate([meta["train"][0], _flatten(bundle.x_train), ztr], axis=1), np.concatenate([meta["val"][0], _flatten(bundle.x_val), zva], axis=1), np.concatenate([meta["test"][0], _flatten(bundle.x_test), zte], axis=1), meta["train"][1] + bundle.feature_names + znames, "metadata_features,sensor_raw_features,jepa_global_embeddings")
                        if protocol != "sensor_only":
                            add_core(protocol, "metadata_plus_sensor_raw", np.concatenate([meta["train"][0], _flatten(bundle.x_train)], axis=1), np.concatenate([meta["val"][0], _flatten(bundle.x_val)], axis=1), np.concatenate([meta["test"][0], _flatten(bundle.x_test)], axis=1), meta["train"][1] + bundle.feature_names, "metadata_features,sensor_raw_features")

                if bench_cfg.get("include_gbt_feature_value", True):
                    meta = meta_cycle
                    eng_train, eng_names = engineered["train"]
                    eng_val, _ = engineered["val"]
                    eng_test, _ = engineered["test"]
                    for estimator in gbt_estimators:
                        prefix = "gbt" if estimator == "hist_gradient_boosting" else estimator
                        gbt_sets = [
                            (f"{prefix}_metadata_only", meta["train"][0], meta["val"][0], meta["test"][0], meta["train"][1], "metadata_features,cycle_features"),
                            (f"{prefix}_sensor_engineered_only", eng_train, eng_val, eng_test, eng_names, "sensor_engineered_features"),
                            (f"{prefix}_metadata_plus_sensor_engineered", np.concatenate([meta["train"][0], eng_train], axis=1), np.concatenate([meta["val"][0], eng_val], axis=1), np.concatenate([meta["test"][0], eng_test], axis=1), meta["train"][1] + eng_names, "metadata_features,cycle_features,sensor_engineered_features"),
                        ]
                        if "current_z" in world:
                            ztr, zva, zte, znames = world["current_z"]
                            pftr, pfva, pfte, pfnames = world["predicted_future_z"]
                            wstr, wsva, wste, wnames = world["world_model_score"]
                            gbt_sets.extend(
                                [
                                    (f"{prefix}_current_z_only", ztr, zva, zte, znames, "jepa_global_embeddings"),
                                    (f"{prefix}_predicted_future_z_only", pftr, pfva, pfte, pfnames, "world_model_features"),
                                    (f"{prefix}_metadata_plus_current_z", np.concatenate([meta["train"][0], ztr], axis=1), np.concatenate([meta["val"][0], zva], axis=1), np.concatenate([meta["test"][0], zte], axis=1), meta["train"][1] + znames, "metadata_features,cycle_features,jepa_global_embeddings"),
                                    (f"{prefix}_metadata_plus_predicted_future_z", np.concatenate([meta["train"][0], pftr], axis=1), np.concatenate([meta["val"][0], pfva], axis=1), np.concatenate([meta["test"][0], pfte], axis=1), meta["train"][1] + pfnames, "metadata_features,cycle_features,world_model_features"),
                                    (f"{prefix}_metadata_plus_current_z_plus_world_score", np.concatenate([meta["train"][0], ztr, wstr], axis=1), np.concatenate([meta["val"][0], zva, wsva], axis=1), np.concatenate([meta["test"][0], zte, wste], axis=1), meta["train"][1] + znames + wnames, "metadata_features,cycle_features,jepa_global_embeddings,world_model_features"),
                                    (f"{prefix}_metadata_plus_engineered_plus_jepa", np.concatenate([meta["train"][0], eng_train, ztr], axis=1), np.concatenate([meta["val"][0], eng_val, zva], axis=1), np.concatenate([meta["test"][0], eng_test, zte], axis=1), meta["train"][1] + eng_names + znames, "metadata_features,cycle_features,sensor_engineered_features,jepa_global_embeddings"),
                                ]
                            )
                        for model_name, xtr, xva, xte, names, groups in gbt_sets:
                            _run_feature_row(bundle, all_rows, all_tool_rows, importance_rows, xtr, xva, xte, names, model_name, "gbt_feature_value", "operational", groups, estimator, seed, horizon, target, permutation=permutation)

    results = add_delta_vs_metadata(pd.DataFrame(all_rows), baseline_model="metadata_only")
    tool_results = pd.DataFrame(all_tool_rows)
    protocol_summary = summarize_by_protocol(results)
    horizon_summary = summarize_by_horizon_target(results)
    mean_std_metrics = [
        c
        for c in [
            "AUROC",
            "AUPRC",
            "precision_at_10pct",
            "recall_at_10pct",
            "false_alarms_per_tool",
            "mean_lead_time",
            "delta_AUPRC_vs_metadata_only",
            "delta_AUROC_vs_metadata_only",
            "delta_Precision@10_vs_metadata_only",
            "delta_Recall@10_vs_metadata_only",
            "delta_false_alarms_vs_metadata_only",
            "delta_lead_time_vs_metadata_only",
        ]
        if c in results.columns
    ]
    mean_std_group = [c for c in ["protocol", "model_name", "model_family", "estimator", "forecast_horizon", "failure_horizon_cycles"] if c in results.columns]
    mean_std = (
        results.groupby(mean_std_group, dropna=False)[mean_std_metrics].agg(["mean", "std", "count"]).reset_index()
        if mean_std_group and mean_std_metrics
        else pd.DataFrame()
    )
    if not mean_std.empty:
        mean_std.columns = ["_".join(str(part) for part in col if part) for col in mean_std.columns.to_flat_index()]
    gbt_results = results[results["model_name"].astype(str).str.startswith(("gbt_", "xgboost_", "lightgbm_"))].copy()
    deltas = results.copy()
    hard_split_results = pd.DataFrame()
    calibration_protocol = {
        "threshold": "Chosen on validation by best F1; test is not used for threshold selection.",
        "calibration": "No test calibration. Probabilistic calibration is measured by Brier/ECE.",
        "comparison_rule": "Deltas are computed only within identical protocol, seed, forecast_horizon and failure_horizon_cycles.",
        "official_baselines": availability_as_dict(),
    }
    leakage = {"checks": leakage_checks, "passes": all(check["passes"] for check in leakage_checks)}

    paths = {
        "incremental_results": out_root / "incremental_results.csv",
        "incremental_summary": out_root / "incremental_summary.md",
        "delta_vs_metadata": out_root / "delta_vs_metadata.csv",
        "results_by_tool": out_root / "results_by_tool.csv",
        "results_by_protocol": out_root / "results_by_protocol.csv",
        "results_mean_std": out_root / "results_mean_std.csv",
        "results_by_horizon_target": out_root / "results_by_horizon_target.csv",
        "hard_split_results": out_root / "hard_split_results.csv",
        "calibration_protocol": out_root / "calibration_protocol.json",
        "incremental_value_report": out_root / "incremental_value_report.md",
        "gbt_feature_value_results": out_root / "gbt_feature_value_results.csv",
        "gbt_feature_importance": out_root / "gbt_feature_importance.csv",
        "permutation_importance": out_root / "permutation_importance.csv",
        "feature_group_ablation": out_root / "feature_group_ablation.csv",
        "gbt_vs_jepa_feature_report": out_root / "gbt_vs_jepa_feature_report.md",
        "leakage_report": out_root / "leakage_report.json",
    }
    results.to_csv(paths["incremental_results"], index=False)
    deltas.to_csv(paths["delta_vs_metadata"], index=False)
    tool_results.to_csv(paths["results_by_tool"], index=False)
    protocol_summary.to_csv(paths["results_by_protocol"], index=False)
    mean_std.to_csv(paths["results_mean_std"], index=False)
    horizon_summary.to_csv(paths["results_by_horizon_target"], index=False)
    hard_split_results.to_csv(paths["hard_split_results"], index=False)
    write_json(paths["calibration_protocol"], calibration_protocol)
    write_json(paths["leakage_report"], leakage)
    gbt_results.to_csv(paths["gbt_feature_value_results"], index=False)
    importance_df = pd.DataFrame(importance_rows)
    native_importance = importance_df[importance_df["importance_type"].eq("native")] if not importance_df.empty else pd.DataFrame()
    perm_importance = importance_df[importance_df["importance_type"].str.contains("permutation", na=False)] if not importance_df.empty else pd.DataFrame()
    native_importance.to_csv(paths["gbt_feature_importance"], index=False)
    perm_importance.to_csv(paths["permutation_importance"], index=False)
    feature_group_ablation(results).to_csv(paths["feature_group_ablation"], index=False)

    top = results.sort_values("AUPRC", ascending=False).head(20) if not results.empty else pd.DataFrame()
    metadata = results[results["model_name"].eq("metadata_only")]
    current = results[results["model_name"].eq("metadata_plus_current_z")]
    pred = results[results["model_name"].eq("metadata_plus_predicted_future_z")]
    wm = results[results["model_name"].eq("metadata_plus_world_model_score")]
    def _mean_delta(df: pd.DataFrame, col: str) -> str:
        if df.empty or col not in df:
            return "pending"
        return f"{float(df[col].mean()):+.4f}"

    write_markdown_report(
        paths["incremental_summary"],
        "Incremental Sensor Value Summary",
        {
            "Direct Answer": (
                "This benchmark measures value over metadata/cycle. A high absolute AUPRC is not treated as Sensor-JEPA evidence unless "
                "`delta_AUPRC_vs_metadata_only` is positive in the same protocol/seed/h/K."
            ),
            "Leakage": f"`{json.dumps({'passes': leakage['passes']})}`",
            "Metadata Baseline Rows": str(len(metadata)),
            "Metadata Plus Current Z Mean Delta AUPRC": _mean_delta(current, "delta_AUPRC_vs_metadata_only"),
            "Metadata Plus Predicted Future Z Mean Delta AUPRC": _mean_delta(pred, "delta_AUPRC_vs_metadata_only"),
            "Metadata Plus World Model Score Mean Delta AUPRC": _mean_delta(wm, "delta_AUPRC_vs_metadata_only"),
            "Top Rows": markdown_table(top.to_dict("records")) if len(top) else "No rows.",
        },
    )
    write_markdown_report(
        paths["incremental_value_report"],
        "Incremental Value Report",
        {
            "Protocol": "Operational allows metadata/cycle. No-cycle removes explicit cycle features. Sensor-only removes metadata.",
            "Answers": "\n".join(
                [
                    f"- Metadata/cycle dominance is assessed by metadata-only rows and their deltas.",
                    f"- Current z delta AUPRC vs metadata-only: {_mean_delta(current, 'delta_AUPRC_vs_metadata_only')}.",
                    f"- Predicted future z delta AUPRC vs metadata-only: {_mean_delta(pred, 'delta_AUPRC_vs_metadata_only')}.",
                    f"- World model score delta AUPRC vs metadata-only: {_mean_delta(wm, 'delta_AUPRC_vs_metadata_only')}.",
                    "- SOTA claim remains false unless official baselines and stable positive deltas are present.",
                ]
            ),
            "Top Rows": markdown_table(top.to_dict("records")) if len(top) else "No rows.",
        },
    )
    gbt_top = gbt_results.sort_values("AUPRC", ascending=False).head(20) if not gbt_results.empty else pd.DataFrame()
    write_markdown_report(
        paths["gbt_vs_jepa_feature_report"],
        "GBT vs JEPA Feature Value Report",
        {
            "Direct Answer": "This report compares GBT metadata-only, engineered sensor features, JEPA embeddings, and combinations. DenseSensorJEPA rows are pending unless a dense checkpoint/features file is added to the config.",
            "Official Baselines": json.dumps(availability_as_dict(), indent=2),
            "Feature Importance": "Native feature importance is saved when the estimator exposes it. Permutation importance is optional and off by default.",
            "Top GBT Rows": markdown_table(gbt_top.to_dict("records")) if len(gbt_top) else "No GBT rows.",
        },
    )
    return paths
