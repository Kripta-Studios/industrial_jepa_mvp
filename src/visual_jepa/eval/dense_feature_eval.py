from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score


def quantile_threshold(scores: np.ndarray, q: float = 0.99) -> float:
    scores = np.asarray(scores, dtype=float)
    if len(scores) == 0:
        return 0.0
    return float(np.quantile(scores, q))


def binary_operating_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float, prefix: str = "") -> dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    pred = (scores >= threshold).astype(int)
    out: dict[str, Any] = {}
    k = lambda name: f"{prefix}{name}" if prefix else name
    try:
        out[k("AUROC")] = float(roc_auc_score(y_true, scores))
    except Exception:
        out[k("AUROC")] = None
    try:
        out[k("AUPRC")] = float(average_precision_score(y_true, scores))
    except Exception:
        out[k("AUPRC")] = None
    out[k("F1")] = float(f1_score(y_true, pred, zero_division=0))
    try:
        out[k("balanced_accuracy")] = float(balanced_accuracy_score(y_true, pred))
    except Exception:
        out[k("balanced_accuracy")] = None
    fp = int(((pred == 1) & (y_true == 0)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    out[k("false_positive_rate")] = float(fp / max(fp + tn, 1))
    out[k("threshold")] = float(threshold)
    out[k("tp")] = tp
    out[k("fp")] = fp
    out[k("tn")] = tn
    out[k("fn")] = fn
    return out


def pixel_overlap_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float, prefix: str = "pixel_") -> dict[str, Any]:
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    scores = np.asarray(scores).astype(float).reshape(-1)
    pred = (scores >= threshold).astype(int)
    out = binary_operating_metrics(y_true, scores, threshold, prefix=prefix)
    inter = int(((pred == 1) & (y_true == 1)).sum())
    union = int(((pred == 1) | (y_true == 1)).sum())
    out[f"{prefix}IoU"] = float(inter / max(union, 1))
    out[f"{prefix}Dice"] = float(2 * inter / max(int(pred.sum()) + int(y_true.sum()), 1))
    return out
