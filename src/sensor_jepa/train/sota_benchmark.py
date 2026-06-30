from __future__ import annotations

import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from common.config import get_device_name
from common.forecast_metrics import (
    forecast_metrics,
    metrics_by_tool,
    save_forecast_plots,
    threshold_from_validation,
)
from common.paths import ensure_dir
from common.reports import markdown_table, write_json, write_markdown_report
from common.seed import seed_everything
from sensor_jepa.data.cnc_world_model import TransitionBundle, prepare_transition_from_config
from sensor_jepa.models.sensor_world_model import ActionConditionedSensorWorldModel
from sensor_jepa.models.strong_baselines import (
    CNN1DClassifier,
    GRUClassifier,
    TCNClassifier,
    predict_scores,
    run_matrix_baselines,
    run_official_rocket_baselines,
    run_rocket_lite_baseline,
    run_tabular_baselines,
    run_ts2vec_proxy,
    train_torch_binary_classifier,
)
from sensor_jepa.train.world_model import build_world_model_from_config


FORBIDDEN_PATTERNS = [
    "cycletofailure",
    "cycle_to_failure",
    "cycletofailurenormalized",
    "rul",
    "remaining_life",
    "remaining_cycles",
    "failure_soon",
    "future_failure",
    "target",
    "label",
    "life_stage",
    "wear_class",
]


def _cfg_for_combo(cfg: dict[str, Any], seed: int, horizon: int, target: int, out_root: Path) -> dict[str, Any]:
    combo = deepcopy(cfg)
    combo["seed"] = seed
    combo.setdefault("world_model", {})
    combo["world_model"]["forecast_horizon"] = horizon
    combo["world_model"]["failure_horizon_cycles"] = target
    combo["outputs"] = deepcopy(combo.get("outputs", {}))
    combo["outputs"]["root"] = str(out_root / f"seed_{seed}" / f"h_{horizon}" / f"k_{target}")
    return combo


def leakage_report(bundle: TransitionBundle) -> dict[str, Any]:
    train_tools = set(bundle.train_meta["ToolIndex"].astype(int).unique().tolist()) if len(bundle.train_meta) else set()
    val_tools = set(bundle.val_meta["ToolIndex"].astype(int).unique().tolist()) if len(bundle.val_meta) else set()
    test_tools = set(bundle.test_meta["ToolIndex"].astype(int).unique().tolist()) if len(bundle.test_meta) else set()
    overlap = {
        "train_val": sorted(train_tools & val_tools),
        "train_test": sorted(train_tools & test_tools),
        "val_test": sorted(val_tools & test_tools),
    }
    no_overlap = not any(overlap.values())
    return {
        "split_type": "by_tool",
        "train_tools": sorted(train_tools),
        "val_tools": sorted(val_tools),
        "test_tools": sorted(test_tools),
        "overlap": overlap,
        "passes": no_overlap,
        "note": "Forecasting windows are generated inside each tool, then split by disjoint tools.",
    }


def feature_audit(bundle: TransitionBundle) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(location: str, column: str, used_by_core_model: bool, role: str) -> None:
        lower = column.lower().replace(" ", "").replace("-", "_")
        matches = [p for p in FORBIDDEN_PATTERNS if p in lower]
        cycle_proxy = lower in {"numberofcycle", "source_cycle", "cycle", "cycle_id", "cycleindex", "cycle_index"}
        rows.append(
            {
                "location": location,
                "column": column,
                "role": role,
                "used_by_core_model": used_by_core_model,
                "forbidden_match": ",".join(matches),
                "is_forbidden": bool(matches),
                "is_cycle_proxy": cycle_proxy,
            }
        )

    for col in bundle.feature_names:
        add("encoder_features", col, True, "sensor_feature")
    for col in bundle.action_names:
        add("action_vector", col, True, "action_or_context")
    for split in ["train", "val", "test"]:
        meta = getattr(bundle, f"{split}_meta")
        for col in meta.columns:
            add(f"{split}_metadata", col, False, "metadata_or_target")
    audit = pd.DataFrame(rows).drop_duplicates(["location", "column", "role"]).reset_index(drop=True)
    used_forbidden = audit[(audit["used_by_core_model"]) & (audit["is_forbidden"])]
    report = {
        "passes": len(used_forbidden) == 0,
        "forbidden_patterns": FORBIDDEN_PATTERNS,
        "used_forbidden_columns": used_forbidden.to_dict("records"),
        "cycle_proxy_note": "Cycle proxies are excluded from core model features/actions and evaluated only in cycle-only baselines.",
        "encoder_feature_count": len(bundle.feature_names),
        "action_count": len(bundle.action_names),
    }
    return audit, report


def _meta_arrays(bundle: TransitionBundle, split: str) -> tuple[np.ndarray, np.ndarray]:
    meta = getattr(bundle, f"{split}_meta")
    tools = meta["ToolIndex"].to_numpy() if len(meta) and "ToolIndex" in meta else np.array([])
    ctf = meta["CycleToFailure"].to_numpy() if len(meta) and "CycleToFailure" in meta else np.array([])
    return tools, ctf


def _cycle_ids(bundle: TransitionBundle, split: str) -> np.ndarray:
    meta = getattr(bundle, f"{split}_meta")
    return meta["NumberOfCycle"].to_numpy() if len(meta) and "NumberOfCycle" in meta else np.arange(len(getattr(bundle, f"y_failure_{split}")))


def _row_from_scores(
    bundle: TransitionBundle,
    model_name: str,
    model_family: str,
    val_scores: np.ndarray,
    test_scores: np.ndarray,
    seed: int,
    horizon: int,
    target: int,
    train_time_sec: float,
    notes: str = "",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    threshold = threshold_from_validation(bundle.y_failure_val, val_scores)
    test_tools, test_ctf = _meta_arrays(bundle, "test")
    row = forecast_metrics(
        bundle.y_failure_test,
        test_scores,
        threshold=threshold,
        tool_ids=test_tools,
        cycle_to_failure=test_ctf,
    )
    row.update(
        {
            "dataset": "cnc_milling",
            "task": "failure_soon_prediction",
            "model_name": model_name,
            "model_family": model_family,
            "seed": seed,
            "forecast_horizon": horizon,
            "failure_horizon_cycles": target,
            "threshold_source": "validation_best_f1",
            "train_time_sec": train_time_sec,
            "test_failure_rate": float(np.mean(bundle.y_failure_test)) if len(bundle.y_failure_test) else 0.0,
            "notes": notes,
        }
    )
    tool_rows = metrics_by_tool(bundle.y_failure_test, test_scores, threshold, test_tools)
    for r in tool_rows:
        r.update({"model_name": model_name, "seed": seed, "forecast_horizon": horizon, "failure_horizon_cycles": target})
    return row, tool_rows


@torch.no_grad()
def _encode_current(model: ActionConditionedSensorWorldModel, x: np.ndarray, device: str, batch_size: int = 256) -> np.ndarray:
    outs = []
    for i in range(0, len(x), batch_size):
        xb = torch.tensor(x[i : i + batch_size], dtype=torch.float32, device=device)
        outs.append(model.encoder(xb).cpu().numpy())
    return np.concatenate(outs)


@torch.no_grad()
def _predict_future(model: ActionConditionedSensorWorldModel, x: np.ndarray, a: np.ndarray, device: str, batch_size: int = 256) -> np.ndarray:
    outs = []
    for i in range(0, len(x), batch_size):
        xb = torch.tensor(x[i : i + batch_size], dtype=torch.float32, device=device)
        ab = torch.tensor(a[i : i + batch_size], dtype=torch.float32, device=device)
        outs.append(model.predict_next_embedding(xb, ab).cpu().numpy())
    return np.concatenate(outs)


def _train_world_model(
    cfg: dict[str, Any],
    bundle: TransitionBundle,
    device: str,
    use_actions: bool,
    pretrained_encoder: bool,
) -> tuple[ActionConditionedSensorWorldModel, float]:
    seed_everything(int(cfg.get("seed", 42)))
    action_dim = bundle.action_dim if use_actions else 1
    model = build_world_model_from_config(cfg, bundle.input_channels, action_dim).to(device)
    if pretrained_encoder:
        ckpt_path = Path("outputs/sensor_jepa/demo_quick/checkpoints/sensor_jepa.pt")
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            encoder_state = {k.replace("encoder.", ""): v for k, v in ckpt["model_state"].items() if k.startswith("encoder.")}
            model.encoder.load_state_dict(encoder_state, strict=False)
    wm_cfg = cfg.get("world_model", {})
    train_cfg = cfg.get("training", {})
    epochs = int(wm_cfg.get("epochs", train_cfg.get("pretrain_epochs", 8)))
    batch_size = int(train_cfg.get("batch_size", 64))
    actions = bundle.a_train if use_actions else np.zeros((len(bundle.a_train), 1), dtype=np.float32)
    loader = DataLoader(
        TensorDataset(
            torch.tensor(bundle.x_train, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.float32),
            torch.tensor(bundle.x_next_train, dtype=torch.float32),
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=float(wm_cfg.get("learning_rate", 1e-3)), weight_decay=float(train_cfg.get("weight_decay", 1e-4)))
    start = time.time()
    for _ in range(epochs):
        model.train()
        for xb, ab, xnb in loader:
            xb, ab, xnb = xb.to(device), ab.to(device), xnb.to(device)
            opt.zero_grad(set_to_none=True)
            out = model(xb, ab, xnb)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
    model.eval()
    return model, time.time() - start


def _fit_probe_scores(z_train, y_train, z_val, z_test, seed: int, calibrated: bool = False, y_val: np.ndarray | None = None):
    base = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed)
    if calibrated:
        if y_val is None:
            raise ValueError("y_val is required for validation-only calibration.")
        base_clf = make_pipeline(StandardScaler(), base)
        base_clf.fit(z_train, y_train)
        val_raw = predict_scores(base_clf, z_val).reshape(-1, 1)
        test_raw = predict_scores(base_clf, z_test).reshape(-1, 1)
        if len(np.unique(y_val)) < 2:
            return val_raw[:, 0], test_raw[:, 0]
        calibrator = LogisticRegression(max_iter=1000, random_state=seed)
        calibrator.fit(val_raw, y_val)
        return calibrator.predict_proba(val_raw)[:, 1], calibrator.predict_proba(test_raw)[:, 1]
    clf = make_pipeline(StandardScaler(), base)
    clf.fit(z_train, y_train)
    return predict_scores(clf, z_val), predict_scores(clf, z_test)


def run_combo(cfg: dict[str, Any], include_slow: bool = True) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    seed = int(cfg["seed"])
    horizon = int(cfg["world_model"]["forecast_horizon"])
    target = int(cfg["world_model"]["failure_horizon_cycles"])
    device = get_device_name(cfg.get("device", "auto"))
    bundle = prepare_transition_from_config(cfg)
    leak = leakage_report(bundle)
    rows: list[dict[str, Any]] = []
    tool_rows: list[dict[str, Any]] = []

    for candidate in run_tabular_baselines(bundle.x_train, bundle.y_failure_train, bundle.x_val, bundle.y_failure_val, bundle.x_test, seed):
        row, by_tool = _row_from_scores(
            bundle,
            candidate["model_name"],
            candidate["model_family"],
            candidate["val_scores"],
            candidate["test_scores"],
            seed,
            horizon,
            target,
            candidate["train_time_sec"],
            candidate.get("notes", ""),
        )
        rows.append(row)
        tool_rows.extend(by_tool)

    adversarial_sets = [
        (
            "action_only",
            bundle.a_train,
            bundle.a_val,
            bundle.a_test,
            "adversarial_metadata",
            "only physical/process action vector; no sensors",
            ("logistic_regression", "random_forest", "hist_gradient_boosting"),
        ),
        (
            "cycle_index_only",
            _metadata_matrix(bundle, "train", ["source_cycle"]),
            _metadata_matrix(bundle, "val", ["source_cycle"]),
            _metadata_matrix(bundle, "test", ["source_cycle"]),
            "adversarial_cycle",
            "only current/source cycle index; no sensors/actions",
            ("logistic_regression", "hist_gradient_boosting"),
        ),
        (
            "metadata_only_no_sensor",
            np.concatenate([bundle.a_train, _metadata_matrix(bundle, "train", ["source_cycle"])], axis=1),
            np.concatenate([bundle.a_val, _metadata_matrix(bundle, "val", ["source_cycle"])], axis=1),
            np.concatenate([bundle.a_test, _metadata_matrix(bundle, "test", ["source_cycle"])], axis=1),
            "adversarial_metadata",
            "actions plus source cycle; no sensor windows",
            ("logistic_regression", "random_forest", "hist_gradient_boosting"),
        ),
        (
            "sensor_plus_actions",
            np.concatenate([bundle.x_train.reshape(len(bundle.x_train), -1), bundle.a_train], axis=1),
            np.concatenate([bundle.x_val.reshape(len(bundle.x_val), -1), bundle.a_val], axis=1),
            np.concatenate([bundle.x_test.reshape(len(bundle.x_test), -1), bundle.a_test], axis=1),
            "adversarial_sensor_plus_metadata",
            "current sensor window plus action vector",
            ("logistic_regression", "hist_gradient_boosting"),
        ),
        (
            "sensor_only_no_actions",
            bundle.x_train,
            bundle.x_val,
            bundle.x_test,
            "adversarial_sensor_only",
            "current sensor window only; explicit no-actions baseline",
            ("logistic_regression", "hist_gradient_boosting"),
        ),
    ]
    for prefix, xtr, xva, xte, family, notes, model_names in adversarial_sets:
        for candidate in run_matrix_baselines(xtr, bundle.y_failure_train, xva, xte, seed, prefix, family, notes, model_names=model_names):
            row, by_tool = _row_from_scores(
                bundle,
                candidate["model_name"],
                candidate["model_family"],
                candidate["val_scores"],
                candidate["test_scores"],
                seed,
                horizon,
                target,
                candidate["train_time_sec"],
                candidate.get("notes", ""),
            )
            rows.append(row)
            tool_rows.extend(by_tool)

    for name, model in [
        ("cnn1d", CNN1DClassifier(bundle.input_channels)),
        ("gru", GRUClassifier(bundle.input_channels)),
        ("tcn", TCNClassifier(bundle.input_channels)),
    ]:
        out = train_torch_binary_classifier(
            model,
            bundle.x_train,
            bundle.y_failure_train,
            bundle.x_val,
            bundle.x_test,
            device=device,
            seed=seed,
            epochs=int(cfg.get("sota_benchmark", {}).get("deep_epochs", 6)),
        )
        row, by_tool = _row_from_scores(bundle, name, "deep_supervised", out["val_scores"], out["test_scores"], seed, horizon, target, out["train_time_sec"], "current_window_raw")
        rows.append(row)
        tool_rows.extend(by_tool)

    for multi in [False, True]:
        out = run_rocket_lite_baseline(bundle.x_train, bundle.y_failure_train, bundle.x_val, bundle.x_test, seed=seed, multi=multi)
        row, by_tool = _row_from_scores(bundle, out["model_name"], out["model_family"], out["val_scores"], out["test_scores"], seed, horizon, target, out["train_time_sec"], out["notes"])
        rows.append(row)
        tool_rows.extend(by_tool)

    for out in run_official_rocket_baselines(bundle.x_train, bundle.y_failure_train, bundle.x_val, bundle.x_test, seed=seed):
        row, by_tool = _row_from_scores(bundle, out["model_name"], out["model_family"], out["val_scores"], out["test_scores"], seed, horizon, target, out["train_time_sec"], out["notes"])
        rows.append(row)
        tool_rows.extend(by_tool)

    if include_slow:
        out = run_ts2vec_proxy(
            bundle.x_train,
            bundle.y_failure_train,
            bundle.x_val,
            bundle.x_test,
            device=device,
            seed=seed,
            epochs=int(cfg.get("sota_benchmark", {}).get("ssl_epochs", 6)),
        )
        row, by_tool = _row_from_scores(bundle, out["model_name"], out["model_family"], out["val_scores"], out["test_scores"], seed, horizon, target, out["train_time_sec"], out["notes"])
        rows.append(row)
        tool_rows.extend(by_tool)

    for use_actions, pretrained in [(True, False), (False, False), (True, True)]:
        model, train_sec = _train_world_model(cfg, bundle, device, use_actions=use_actions, pretrained_encoder=pretrained)
        suffix = "actions" if use_actions else "no_actions"
        init = "jepa_init" if pretrained else "scratch"
        a_train = bundle.a_train if use_actions else np.zeros((len(bundle.a_train), 1), dtype=np.float32)
        a_val = bundle.a_val if use_actions else np.zeros((len(bundle.a_val), 1), dtype=np.float32)
        a_test = bundle.a_test if use_actions else np.zeros((len(bundle.a_test), 1), dtype=np.float32)

        z_train_current = _encode_current(model, bundle.x_train, device)
        z_val_current = _encode_current(model, bundle.x_val, device)
        z_test_current = _encode_current(model, bundle.x_test, device)
        val_scores, test_scores = _fit_probe_scores(z_train_current, bundle.y_failure_train, z_val_current, z_test_current, seed)
        row, by_tool = _row_from_scores(bundle, f"world_model_current_z_{suffix}_{init}", "jepa_world_model_ablation", val_scores, test_scores, seed, horizon, target, train_sec, "z_t_to_failure_soon")
        rows.append(row)
        tool_rows.extend(by_tool)

        z_train_future = _predict_future(model, bundle.x_train, a_train, device)
        z_val_future = _predict_future(model, bundle.x_val, a_val, device)
        z_test_future = _predict_future(model, bundle.x_test, a_test, device)
        val_scores, test_scores = _fit_probe_scores(z_train_future, bundle.y_failure_train, z_val_future, z_test_future, seed)
        row, by_tool = _row_from_scores(bundle, f"world_model_pred_future_{suffix}_{init}", "jepa_world_model", val_scores, test_scores, seed, horizon, target, train_sec, "predicted_z_t_plus_h_to_failure_soon")
        rows.append(row)
        tool_rows.extend(by_tool)

        if use_actions and not pretrained:
            val_scores, test_scores = _fit_probe_scores(z_train_future, bundle.y_failure_train, z_val_future, z_test_future, seed, calibrated=True, y_val=bundle.y_failure_val)
            row, by_tool = _row_from_scores(bundle, "world_model_pred_future_actions_calibrated", "jepa_world_model", val_scores, test_scores, seed, horizon, target, train_sec, "calibrated_sigmoid_probe")
            rows.append(row)
            tool_rows.extend(by_tool)

            plot_dir = Path(cfg["outputs"]["root"]) / "plots" / "world_model_pred_future_actions"
            save_forecast_plots(
                bundle.y_failure_test,
                test_scores,
                plot_dir,
                f"World model h={horizon} K={target} seed={seed}",
                tool_ids=_meta_arrays_safe(bundle, "test", "ToolIndex"),
                cycle_ids=_cycle_arrays_safe(bundle, "test"),
                threshold=threshold_from_validation(bundle.y_failure_val, val_scores),
                cycle_to_failure=_meta_arrays_safe(bundle, "test", "CycleToFailure"),
            )

    return rows, tool_rows, leak


def _meta_arrays_safe(bundle: TransitionBundle, split: str, col: str) -> np.ndarray:
    meta = getattr(bundle, f"{split}_meta")
    return meta[col].to_numpy() if len(meta) and col in meta else np.array([])


def _cycle_arrays_safe(bundle: TransitionBundle, split: str) -> np.ndarray:
    meta = getattr(bundle, f"{split}_meta")
    return meta["NumberOfCycle"].to_numpy() if len(meta) and "NumberOfCycle" in meta else np.arange(len(getattr(bundle, f"y_failure_{split}")))


def _metadata_matrix(bundle: TransitionBundle, split: str, columns: list[str]) -> np.ndarray:
    meta = getattr(bundle, f"{split}_meta")
    if not len(meta):
        return np.empty((0, len(columns)), dtype=np.float32)
    values = []
    for col in columns:
        if col in meta.columns:
            values.append(pd.to_numeric(meta[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32))
        else:
            values.append(np.zeros(len(meta), dtype=np.float32))
    return np.stack(values, axis=1)


def summarize_results(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    metric = "AUPRC"
    group_cols = ["model_name", "model_family", "forecast_horizon", "failure_horizon_cycles"]
    agg = df.groupby(group_cols, dropna=False).agg(
        AUROC_mean=("AUROC", "mean"),
        AUROC_std=("AUROC", "std"),
        AUPRC_mean=("AUPRC", "mean"),
        AUPRC_std=("AUPRC", "std"),
        precision_at_10pct_mean=("precision_at_10pct", "mean"),
        recall_at_10pct_mean=("recall_at_10pct", "mean"),
        ECE_mean=("ECE", "mean"),
        seeds=("seed", "nunique"),
    ).reset_index()
    agg = agg.sort_values(["AUPRC_mean", "AUROC_mean"], ascending=False)
    agg.insert(0, "rank", range(1, len(agg) + 1))
    return agg


def conclusion_from_summary(summary: pd.DataFrame, leakage_passes: bool) -> str:
    if summary.empty:
        return "No benchmark rows were produced."
    best = summary.iloc[0]
    names = summary["model_name"].astype(str)
    wm_cal = summary[names.eq("world_model_pred_future_actions_calibrated")]
    wm_pred_actions = summary[names.eq("world_model_pred_future_actions_scratch")]
    rocket = summary[summary["model_name"].astype(str).str.contains("rocket", regex=False)]
    ts = summary[summary["model_name"].astype(str).str.contains("ts2vec", regex=False)]
    lines = [
        f"Best model by mean AUPRC: `{best['model_name']}` with AUPRC={best['AUPRC_mean']:.4f}.",
        f"Leakage check passes: `{leakage_passes}`.",
    ]
    if len(wm_pred_actions):
        best_wm_pred = wm_pred_actions.sort_values("AUPRC_mean", ascending=False).iloc[0]
        lines.append(f"Best predicted-future action world model AUPRC={best_wm_pred['AUPRC_mean']:.4f}, AUROC={best_wm_pred['AUROC_mean']:.4f}.")
    if len(wm_cal):
        best_wm_cal = wm_cal.sort_values("AUPRC_mean", ascending=False).iloc[0]
        lines.append(f"Best calibrated predicted-future action world model AUPRC={best_wm_cal['AUPRC_mean']:.4f}, AUROC={best_wm_cal['AUROC_mean']:.4f}.")
    if len(rocket):
        best_rocket = rocket.sort_values("AUPRC_mean", ascending=False).iloc[0]
        lines.append(f"Best Rocket-family baseline AUPRC={best_rocket['AUPRC_mean']:.4f}.")
    if len(ts):
        best_ts = ts.sort_values("AUPRC_mean", ascending=False).iloc[0]
        lines.append(f"TS2Vec proxy AUPRC={best_ts['AUPRC_mean']:.4f}.")
    lines.append("Do not claim SOTA unless full seeds, exact MiniROCKET/MultiROCKET or accepted equivalents, TS2Vec/proper SSL baseline, and literature comparisons are complete.")
    return "\n\n".join(lines)


def _best_row(summary: pd.DataFrame, mask: pd.Series) -> pd.Series | None:
    subset = summary[mask]
    if subset.empty:
        return None
    return subset.sort_values(["AUPRC_mean", "AUROC_mean"], ascending=False).iloc[0]


def _format_best(row: pd.Series | None) -> str:
    if row is None:
        return "missing"
    return f"`{row['model_name']}` AUPRC={row['AUPRC_mean']:.4f}, AUROC={row['AUROC_mean']:.4f}"


def validation_report_sections(
    summary: pd.DataFrame,
    tool_results: pd.DataFrame,
    leakage_passes: bool,
    feature_audit_passes: bool,
) -> dict[str, str]:
    if summary.empty:
        return {"Direct Answer": "No benchmark rows were produced."}

    names = summary["model_name"].astype(str)
    families = summary["model_family"].astype(str)
    wm = _best_row(summary, names.eq("world_model_pred_future_actions_calibrated"))
    if wm is None:
        wm = _best_row(summary, names.str.contains("world_model_pred_future_actions", regex=False))
    action = _best_row(summary, names.str.startswith("action_only"))
    metadata = _best_row(summary, names.str.startswith("metadata_only"))
    cycle = _best_row(summary, names.str.startswith("cycle_index_only"))
    sensor_only = _best_row(summary, names.str.startswith("sensor_only"))
    rocket_official = _best_row(summary, families.eq("rocket_official"))
    rocket_any = _best_row(summary, names.str.contains("rocket", regex=False))
    ts_official = _best_row(summary, names.str.contains("ts2vec_official", regex=False))
    ts_any = _best_row(summary, names.str.contains("ts2vec", regex=False))
    current_z = _best_row(summary, names.str.contains("world_model_current_z_actions", regex=False))
    no_actions = _best_row(summary, names.str.contains("world_model_pred_future_no_actions", regex=False))

    def compare(label: str, other: pd.Series | None, missing_note: str = "missing") -> str:
        if wm is None:
            return f"- {label}: no world-model row available."
        if other is None:
            return f"- {label}: {missing_note}."
        delta = float(wm["AUPRC_mean"] - other["AUPRC_mean"])
        verdict = "yes" if delta > 0 else "no"
        return f"- {label}: {verdict}. WM AUPRC={wm['AUPRC_mean']:.4f}; baseline AUPRC={other['AUPRC_mean']:.4f}; delta={delta:+.4f}."

    hgrid = summary[names.eq("world_model_pred_future_actions_calibrated")]
    if hgrid.empty:
        stability = "No calibrated action world-model grid rows were produced."
    else:
        stability = (
            f"Rows={len(hgrid)}, horizons={sorted(hgrid['forecast_horizon'].unique().tolist())}, "
            f"targets={sorted(hgrid['failure_horizon_cycles'].unique().tolist())}, "
            f"AUPRC range={hgrid['AUPRC_mean'].min():.4f}-{hgrid['AUPRC_mean'].max():.4f}."
        )

    if wm is not None and len(tool_results):
        core_tools = tool_results[tool_results["model_name"].eq(wm["model_name"])]
        if len(core_tools):
            worst_tool = core_tools.sort_values("AUPRC", na_position="last").head(1)
            per_tool = markdown_table(worst_tool.to_dict("records"))
        else:
            per_tool = "No per-tool rows for selected world model."
    else:
        per_tool = "No per-tool rows available."

    dynamics_lines = []
    if wm is not None and current_z is not None:
        dynamics_lines.append(compare("predicted future embedding vs current embedding", current_z))
    if wm is not None and no_actions is not None:
        dynamics_lines.append(compare("action-conditioned world model vs no-actions world model", no_actions))
    if not dynamics_lines:
        dynamics_lines.append("- Not enough ablation rows to judge latent dynamics.")

    official_notes = []
    if rocket_official is None:
        official_notes.append("MiniROCKET/MultiROCKET official not present; fallback Rocket-lite rows cannot support a SOTA claim.")
    if ts_official is None:
        official_notes.append("TS2Vec official not present; current TS2Vec row is a proxy if available.")
    if not official_notes:
        official_notes.append("Official Rocket/TS2Vec rows are present in this run.")

    return {
        "Direct Answer": conclusion_from_summary(summary, leakage_passes),
        "Adversarial Checks": "\n".join(
            [
                compare("World model beats action-only", action),
                compare("World model beats metadata-only", metadata),
                compare("World model beats cycle-index-only", cycle),
                compare("World model beats sensor-only", sensor_only),
                compare("World model beats MiniROCKET/MultiROCKET official", rocket_official, "official Rocket baseline missing"),
                compare("World model beats TS2Vec official", ts_official, "official TS2Vec baseline missing"),
                f"- Leakage check passes: `{leakage_passes}`.",
                f"- Feature audit passes: `{feature_audit_passes}`.",
            ]
        ),
        "Available Baselines": "\n".join(
            [
                f"- World model: {_format_best(wm)}",
                f"- Action-only: {_format_best(action)}",
                f"- Metadata-only: {_format_best(metadata)}",
                f"- Cycle-index-only: {_format_best(cycle)}",
                f"- Best Rocket family: {_format_best(rocket_any)}",
                f"- Best TS2Vec family: {_format_best(ts_any)}",
            ]
        ),
        "Horizon Target Stability": stability,
        "Latent Dynamics Evidence": "\n".join(dynamics_lines),
        "Per Tool Robustness": per_tool,
        "SOTA Claim Rules": "\n".join(
            [
                "- SOTA claim remains false in this report.",
                *[f"- {note}" for note in official_notes],
                "- Use this as an internal SOTA-candidate/adversarial-validation report until official baselines and literature protocol comparisons are complete.",
            ]
        ),
    }


def run_sota_benchmark(
    cfg: dict[str, Any],
    seeds: list[int],
    horizons: list[int],
    targets: list[int],
    out_root: str | Path = "outputs/sensor_jepa/sota_benchmark",
    quick: bool = False,
) -> dict[str, Path]:
    out_root = ensure_dir(out_root)
    audit_cfg = _cfg_for_combo(cfg, seeds[0], horizons[0], targets[0], out_root)
    audit_bundle = prepare_transition_from_config(audit_cfg)
    audit_df, forbidden_report = feature_audit(audit_bundle)
    feature_audit_path = out_root / "feature_audit.csv"
    forbidden_report_path = out_root / "forbidden_columns_report.json"
    audit_df.to_csv(feature_audit_path, index=False)
    write_json(forbidden_report_path, forbidden_report)

    all_rows: list[dict[str, Any]] = []
    all_tool_rows: list[dict[str, Any]] = []
    leakage: dict[str, Any] = {"checks": []}
    for seed in seeds:
        for horizon in horizons:
            for target in targets:
                combo_cfg = _cfg_for_combo(cfg, seed, horizon, target, out_root)
                combo_cfg.setdefault("world_model", {})
                combo_cfg["world_model"]["epochs"] = int(combo_cfg["world_model"].get("epochs", 8 if not quick else 4))
                combo_cfg.setdefault("sota_benchmark", {})
                combo_cfg["sota_benchmark"]["deep_epochs"] = 4 if quick else int(combo_cfg["sota_benchmark"].get("deep_epochs", 8))
                combo_cfg["sota_benchmark"]["ssl_epochs"] = 4 if quick else int(combo_cfg["sota_benchmark"].get("ssl_epochs", 8))
                rows, tool_rows, leak = run_combo(combo_cfg, include_slow=not quick)
                leakage["checks"].append({"seed": seed, "forecast_horizon": horizon, "failure_horizon_cycles": target, **leak})
                all_rows.extend(rows)
                all_tool_rows.extend(tool_rows)

    results = pd.DataFrame(all_rows)
    tool_results = pd.DataFrame(all_tool_rows)
    summary = summarize_results(all_rows)
    leakage["passes"] = all(check["passes"] for check in leakage["checks"])

    results_path = out_root / "sota_benchmark_results.csv"
    by_seed_path = out_root / "results_by_seed.csv"
    by_tool_path = out_root / "results_by_tool.csv"
    by_horizon_path = out_root / "results_by_horizon.csv"
    ablation_path = out_root / "ablation_results.csv"
    action_only_path = out_root / "action_only_results.csv"
    cycle_only_path = out_root / "cycle_only_results.csv"
    metadata_only_path = out_root / "metadata_only_results.csv"
    sensor_only_path = out_root / "sensor_only_results.csv"
    sensor_plus_actions_path = out_root / "sensor_plus_actions_results.csv"
    horizon_target_grid_path = out_root / "horizon_target_grid.csv"
    worst_tool_report_path = out_root / "worst_tool_report.md"
    calibration_protocol_path = out_root / "calibration_protocol.json"
    summary_md = out_root / "sota_benchmark_summary.md"
    candidate_report = out_root / "sota_candidate_report.md"
    validation_report = out_root / "sota_validation_report.md"
    leakage_path = out_root / "leakage_report.json"
    summary_json = out_root / "sota_benchmark_summary.json"
    calibration_path = out_root / "calibration_metrics.csv"

    results.to_csv(results_path, index=False)
    results.to_csv(by_seed_path, index=False)
    tool_results.to_csv(by_tool_path, index=False)
    summary.to_csv(by_horizon_path, index=False)
    results[results["model_name"].astype(str).str.contains("world_model", regex=False)].to_csv(ablation_path, index=False)
    results[results["model_name"].astype(str).str.startswith("action_only")].to_csv(action_only_path, index=False)
    results[results["model_name"].astype(str).str.startswith("cycle_index_only")].to_csv(cycle_only_path, index=False)
    results[results["model_name"].astype(str).str.startswith("metadata_only")].to_csv(metadata_only_path, index=False)
    results[results["model_name"].astype(str).str.startswith("sensor_only")].to_csv(sensor_only_path, index=False)
    results[results["model_name"].astype(str).str.startswith("sensor_plus_actions")].to_csv(sensor_plus_actions_path, index=False)

    action_wm = summary[summary["model_name"].eq("world_model_pred_future_actions_calibrated")]
    if len(action_wm):
        grid = action_wm.pivot_table(index="forecast_horizon", columns="failure_horizon_cycles", values="AUPRC_mean")
        grid.to_csv(horizon_target_grid_path)
        plt.figure(figsize=(6, 4))
        plt.imshow(grid.to_numpy(dtype=float), aspect="auto", cmap="viridis")
        plt.colorbar(label="AUPRC mean")
        plt.xticks(range(len(grid.columns)), grid.columns)
        plt.yticks(range(len(grid.index)), grid.index)
        plt.xlabel("Failure horizon K")
        plt.ylabel("Forecast horizon h")
        plt.title("World model AUPRC grid")
        plt.tight_layout()
        plt.savefig(out_root / "horizon_target_heatmap.png", dpi=150)
        plt.close()
    else:
        pd.DataFrame().to_csv(horizon_target_grid_path, index=False)

    best_model = summary.iloc[0]["model_name"] if len(summary) else ""
    worst_rows = tool_results[tool_results["model_name"].eq(best_model)].sort_values("AUPRC", na_position="last").head(20)
    write_markdown_report(
        worst_tool_report_path,
        "Worst Tool Report",
        {
            "Best Model Considered": f"`{best_model}`",
            "Worst Tools": markdown_table(worst_rows.to_dict("records")),
            "Caution": "Tool-level AUROC can be undefined when a tool split contains one class only; AUPRC and false alarms remain more informative.",
        },
    )

    write_json(
        calibration_protocol_path,
        {
            "risk_head": "LogisticRegression on train embeddings.",
            "calibrator": "Platt-style LogisticRegression fitted on validation scores only for calibrated world model.",
            "threshold": "Chosen from validation by best F1.",
            "test_usage": "Test is used only once for final metric computation.",
            "forbidden_columns_report": str(forbidden_report_path),
        },
    )
    calibration_cols = [
        "model_name",
        "seed",
        "forecast_horizon",
        "failure_horizon_cycles",
        "brier_score",
        "ECE",
        "threshold",
        "AUPRC",
        "AUROC",
    ]
    results[[c for c in calibration_cols if c in results.columns]].to_csv(calibration_path, index=False)
    write_json(leakage_path, leakage)
    write_json(
        summary_json,
        {
            "best_model": summary.iloc[0]["model_name"] if len(summary) else None,
            "best_AUPRC_mean": float(summary.iloc[0]["AUPRC_mean"]) if len(summary) else None,
            "leakage_passes": leakage["passes"],
            "feature_audit_passes": forbidden_report["passes"],
            "sota_claim": False,
        },
    )
    write_markdown_report(
        summary_md,
        "Sensor World Model SOTA-Candidate Benchmark Summary",
        {
            "Conclusion": conclusion_from_summary(summary, leakage["passes"]),
            "Ranking": markdown_table(summary.head(25).to_dict("records")),
        },
    )
    write_markdown_report(
        candidate_report,
        "SOTA Candidate Report",
        {
            "Direct Answer": conclusion_from_summary(summary, leakage["passes"]),
            "Required Cautions": (
                "MiniROCKET/MultiROCKET are fallback lite implementations when sktime/aeon are missing. "
                "TS2Vec is a proxy contrastive encoder unless an official implementation is integrated. "
                "Therefore this is a SOTA-candidate benchmark scaffold, not a SOTA claim. "
                f"Feature audit passes: `{forbidden_report['passes']}`."
            ),
            "Top Results": markdown_table(summary.head(15).to_dict("records")),
            "Leakage": f"`{json.dumps({'passes': leakage['passes']})}`",
        },
    )
    write_markdown_report(
        validation_report,
        "SOTA Validation Report",
        validation_report_sections(summary, tool_results, leakage["passes"], forbidden_report["passes"]),
    )
    return {
        "results": results_path,
        "summary": summary_md,
        "candidate_report": candidate_report,
        "validation_report": validation_report,
        "leakage": leakage_path,
        "by_tool": by_tool_path,
        "ablation": ablation_path,
        "feature_audit": feature_audit_path,
        "forbidden_columns": forbidden_report_path,
        "calibration_protocol": calibration_protocol_path,
        "worst_tool_report": worst_tool_report_path,
    }
