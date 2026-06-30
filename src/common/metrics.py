from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_recall_curve,
    r2_score,
    roc_auc_score,
)


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        return float(value)
    except Exception:
        return None


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None = None,
) -> dict[str, Any]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    out: dict[str, Any] = {
        "accuracy": _safe_float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": _safe_float(balanced_accuracy_score(y_true, y_pred)),
        "macro_F1": _safe_float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_F1": _safe_float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    if y_score is not None:
        y_score = np.asarray(y_score)
        try:
            if y_score.ndim == 1 or y_score.shape[1] == 2:
                score = y_score if y_score.ndim == 1 else y_score[:, 1]
                out["AUROC"] = _safe_float(roc_auc_score(y_true, score))
                out["AUPRC"] = _safe_float(average_precision_score(y_true, score))
            else:
                out["AUROC"] = _safe_float(roc_auc_score(y_true, y_score, multi_class="ovr"))
        except Exception:
            out["AUROC"] = None
            out["AUPRC"] = None
    return out


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    out = {
        "MAE": _safe_float(mean_absolute_error(y_true, y_pred)),
        "RMSE": _safe_float(rmse),
        "R2": _safe_float(r2_score(y_true, y_pred)),
    }
    try:
        out["spearman"] = _safe_float(np.corrcoef(np.argsort(y_true), np.argsort(y_pred))[0, 1])
    except Exception:
        out["spearman"] = None
    return out


def anomaly_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float | None = None,
    prefix: str = "",
) -> dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    out: dict[str, Any] = {}
    key = lambda name: f"{prefix}{name}" if prefix else name
    try:
        out[key("AUROC")] = _safe_float(roc_auc_score(y_true, scores))
    except Exception:
        out[key("AUROC")] = None
    try:
        out[key("AUPRC")] = _safe_float(average_precision_score(y_true, scores))
    except Exception:
        out[key("AUPRC")] = None
    if threshold is None:
        threshold = best_f1_threshold(y_true, scores)
    pred = (scores >= threshold).astype(int)
    out[key("threshold")] = _safe_float(threshold)
    out[key("accuracy")] = _safe_float(accuracy_score(y_true, pred))
    out[key("F1")] = _safe_float(f1_score(y_true, pred, zero_division=0))
    return out


def best_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    try:
        precision, recall, thresholds = precision_recall_curve(y_true, scores)
        f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
        if len(f1) == 0:
            return float(np.quantile(scores, 0.95))
        return float(thresholds[int(np.nanargmax(f1))])
    except Exception:
        return float(np.quantile(scores, 0.95))


def flatten_metrics(row: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in row.items() if not isinstance(v, (list, dict))}

