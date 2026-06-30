from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, precision_recall_curve, roc_auc_score


def _safe_metric(fn, default=None):
    try:
        return float(fn())
    except Exception:
        return default


def best_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if len(thresholds) == 0:
        return float(np.quantile(scores, 0.95))
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.nanargmax(f1))])


def precision_recall_at_fraction(y_true: np.ndarray, scores: np.ndarray, fraction: float) -> tuple[float, float]:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    k = max(1, int(np.ceil(len(scores) * fraction)))
    idx = np.argsort(scores)[::-1][:k]
    positives = max(int(y_true.sum()), 1)
    precision = float(y_true[idx].sum() / k)
    recall = float(y_true[idx].sum() / positives)
    return precision, recall


def image_anomaly_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float | None]:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    threshold = best_f1_threshold(y_true, scores)
    pred = (scores >= threshold).astype(int)
    p5, r5 = precision_recall_at_fraction(y_true, scores, 0.05)
    p10, r10 = precision_recall_at_fraction(y_true, scores, 0.10)
    return {
        "image_AUROC": _safe_metric(lambda: roc_auc_score(y_true, scores)),
        "image_AUPRC": _safe_metric(lambda: average_precision_score(y_true, scores)),
        "F1": _safe_metric(lambda: f1_score(y_true, pred, zero_division=0)),
        "balanced_accuracy": _safe_metric(lambda: balanced_accuracy_score(y_true, pred)),
        "Precision@5%": p5,
        "Precision@10%": p10,
        "Recall@5%": r5,
        "Recall@10%": r10,
        "threshold": threshold,
    }


def pixel_anomaly_metrics(mask_flat: np.ndarray, score_flat: np.ndarray) -> dict[str, float | None]:
    mask_flat = np.asarray(mask_flat).astype(int)
    score_flat = np.asarray(score_flat).astype(float)
    out = {
        "pixel_AUROC": _safe_metric(lambda: roc_auc_score(mask_flat, score_flat)),
        "pixel_AUPRC": _safe_metric(lambda: average_precision_score(mask_flat, score_flat)),
        "PRO": None,
    }
    try:
        t = best_f1_threshold(mask_flat, score_flat)
        pred = score_flat >= t
        inter = np.logical_and(pred, mask_flat == 1).sum()
        union = np.logical_or(pred, mask_flat == 1).sum()
        out["IoU"] = float(inter / max(union, 1))
    except Exception:
        out["IoU"] = None
    return out
