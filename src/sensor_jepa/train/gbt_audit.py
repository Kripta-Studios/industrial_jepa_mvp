from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from common.forecast_metrics import forecast_metrics, threshold_from_validation
from common.paths import ensure_dir
from common.reports import markdown_table, write_markdown_report
from sensor_jepa.data.cnc_world_model import prepare_transition_from_config
from sensor_jepa.models.strong_baselines import predict_scores
from sensor_jepa.train.incremental_value_benchmark import _metadata_matrix


def _fit_model(name: str, x_train: np.ndarray, y_train: np.ndarray, seed: int):
    if name == "logistic_regression":
        return make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed)), None
    if name == "hist_gradient_boosting_default":
        return HistGradientBoostingClassifier(random_state=seed), None
    if name == "hist_gradient_boosting_tuned":
        weights = compute_sample_weight("balanced", y_train)
        return (
            HistGradientBoostingClassifier(
                random_state=seed,
                learning_rate=0.03,
                max_iter=500,
                max_leaf_nodes=7,
                l2_regularization=0.01,
                early_stopping=True,
                validation_fraction=0.2,
            ),
            weights,
        )
    if name == "random_forest_tuned":
        return RandomForestClassifier(n_estimators=800, max_depth=4, min_samples_leaf=3, class_weight="balanced", random_state=seed, n_jobs=-1), None
    if name == "xgboost_tuned":
        from xgboost import XGBClassifier

        pos = max(float(y_train.sum()), 1.0)
        neg = max(float(len(y_train) - y_train.sum()), 1.0)
        return (
            XGBClassifier(
                n_estimators=500,
                max_depth=2,
                learning_rate=0.02,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=5.0,
                min_child_weight=2.0,
                scale_pos_weight=neg / pos,
                eval_metric="logloss",
                random_state=seed,
                n_jobs=2,
            ),
            None,
        )
    if name == "lightgbm_tuned":
        from lightgbm import LGBMClassifier

        return (
            LGBMClassifier(
                n_estimators=500,
                learning_rate=0.02,
                num_leaves=7,
                min_child_samples=8,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=5.0,
                class_weight="balanced",
                random_state=seed,
                verbosity=-1,
            ),
            None,
        )
    raise ValueError(f"Unknown GBT audit model: {name}")


def _evaluate(name: str, x_train, y_train, x_val, y_val, x_test, y_test, seed: int) -> dict[str, Any]:
    start = time.time()
    try:
        model, sample_weight = _fit_model(name, x_train, y_train, seed)
        if sample_weight is None:
            model.fit(x_train, y_train)
        else:
            model.fit(x_train, y_train, sample_weight=sample_weight)
        val_scores = predict_scores(model, x_val)
        test_scores = predict_scores(model, x_test)
        threshold = threshold_from_validation(y_val, val_scores)
        row = forecast_metrics(y_test, test_scores, threshold=threshold)
        row.update({"model_name": name, "train_time_sec": time.time() - start, "status": "ok", "notes": ""})
        return row
    except Exception as exc:
        return {"model_name": name, "train_time_sec": time.time() - start, "status": "failed", "notes": f"{type(exc).__name__}: {exc}"}


def _feature_audit(x_train: np.ndarray, x_val: np.ndarray, x_test: np.ndarray, names: list[str]) -> pd.DataFrame:
    rows = []
    for idx, name in enumerate(names):
        train_col = x_train[:, idx]
        rows.append(
            {
                "feature": name,
                "train_nan": int(np.isnan(train_col).sum()),
                "val_nan": int(np.isnan(x_val[:, idx]).sum()),
                "test_nan": int(np.isnan(x_test[:, idx]).sum()),
                "train_std": float(np.nanstd(train_col)),
                "is_constant_train": bool(np.nanstd(train_col) < 1e-12),
                "train_min": float(np.nanmin(train_col)),
                "train_max": float(np.nanmax(train_col)),
            }
        )
    return pd.DataFrame(rows)


def run_gbt_audit(cfg: dict[str, Any], out_dir: str | Path = "outputs/sensor_jepa/incremental_value_benchmark") -> dict[str, Path]:
    out_dir = ensure_dir(out_dir)
    seed = int(cfg.get("seed", 42))
    bundle = prepare_transition_from_config(cfg)
    x_train, names = _metadata_matrix(bundle, "train", include_cycle=True)
    x_val, _ = _metadata_matrix(bundle, "val", include_cycle=True)
    x_test, _ = _metadata_matrix(bundle, "test", include_cycle=True)
    x_train = np.nan_to_num(x_train, nan=0.0, posinf=0.0, neginf=0.0)
    x_val = np.nan_to_num(x_val, nan=0.0, posinf=0.0, neginf=0.0)
    x_test = np.nan_to_num(x_test, nan=0.0, posinf=0.0, neginf=0.0)

    models = [
        "logistic_regression",
        "hist_gradient_boosting_default",
        "hist_gradient_boosting_tuned",
        "random_forest_tuned",
        "xgboost_tuned",
        "lightgbm_tuned",
    ]
    rows = [_evaluate(name, x_train, bundle.y_failure_train, x_val, bundle.y_failure_val, x_test, bundle.y_failure_test, seed) for name in models]
    results = pd.DataFrame(rows).sort_values("AUPRC", ascending=False, na_position="last")
    feature_audit = _feature_audit(x_train, x_val, x_test, names)
    result_path = out_dir / "gbt_audit_results.csv"
    feature_path = out_dir / "gbt_audit_feature_audit.csv"
    report_path = out_dir / "gbt_audit_report.md"
    results.to_csv(result_path, index=False)
    feature_audit.to_csv(feature_path, index=False)

    split_rows = [
        {"split": "train", "rows": len(bundle.y_failure_train), "positives": int(bundle.y_failure_train.sum()), "positive_rate": float(bundle.y_failure_train.mean())},
        {"split": "val", "rows": len(bundle.y_failure_val), "positives": int(bundle.y_failure_val.sum()), "positive_rate": float(bundle.y_failure_val.mean())},
        {"split": "test", "rows": len(bundle.y_failure_test), "positives": int(bundle.y_failure_test.sum()), "positive_rate": float(bundle.y_failure_test.mean())},
    ]
    constant_features = feature_audit[feature_audit["is_constant_train"]]["feature"].tolist()
    logistic = results[results["model_name"].eq("logistic_regression")]
    hgb = results[results["model_name"].eq("hist_gradient_boosting_default")]
    if len(logistic) and len(hgb):
        gap = float(logistic.iloc[0]["AUPRC"] - hgb.iloc[0]["AUPRC"])
        interpretation = (
            f"Default HGB trails logistic by {gap:+.4f} AUPRC on identical metadata rows/features. "
            "Treat default HGB as under-audited until tuned/tree baselines are used."
        )
    else:
        interpretation = "One or more audit models failed; inspect CSV outputs."
    write_markdown_report(
        report_path,
        "GBT Metadata Audit Report",
        {
            "Direct Answer": interpretation,
            "Rows And Target": markdown_table(split_rows),
            "Feature Set": ", ".join(names),
            "Constant Features": ", ".join(constant_features) if constant_features else "None",
            "Model Results": markdown_table(results.to_dict("records")),
            "Feature Audit Output": f"`{feature_path}`",
        },
    )
    return {"results": result_path, "feature_audit": feature_path, "report": report_path}
