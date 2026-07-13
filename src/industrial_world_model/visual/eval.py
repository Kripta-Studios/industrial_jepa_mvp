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


def fit_threshold(
    validation_labels: np.ndarray | None,
    validation_scores: np.ndarray,
    *,
    method: str = "best_f1",
    nominal_quantile: float = 0.99,
) -> float:
    """Fit a decision threshold using validation data only.

    MVTec AD has no official anomalous validation split.  For that setting the
    transparent default is ``nominal_quantile`` on held-out normal training
    images.  ``best_f1`` requires both classes and must never receive test data.
    """

    scores = np.asarray(validation_scores, dtype=float)
    if scores.ndim != 1 or scores.size == 0 or not np.isfinite(scores).all():
        raise ValueError("validation_scores must be a non-empty finite 1-D array")
    if method == "nominal_quantile":
        if not 0.0 < nominal_quantile < 1.0:
            raise ValueError("nominal_quantile must be between 0 and 1")
        if validation_labels is not None and np.any(np.asarray(validation_labels).astype(int) != 0):
            raise ValueError("nominal_quantile expects a nominal-only validation split")
        return float(np.quantile(scores, nominal_quantile))
    if method == "best_f1":
        if validation_labels is None:
            raise ValueError("best_f1 requires validation labels")
        labels = np.asarray(validation_labels).astype(int)
        if labels.shape != scores.shape or np.unique(labels).size < 2:
            raise ValueError("best_f1 requires aligned validation samples with both classes")
        return best_f1_threshold(labels, scores)
    raise ValueError(f"unknown threshold method: {method}")


def precision_recall_at_fraction(y_true: np.ndarray, scores: np.ndarray, fraction: float) -> tuple[float, float]:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    k = max(1, int(np.ceil(len(scores) * fraction)))
    idx = np.argsort(scores)[::-1][:k]
    positives = max(int(y_true.sum()), 1)
    precision = float(y_true[idx].sum() / k)
    recall = float(y_true[idx].sum() / positives)
    return precision, recall


def image_anomaly_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    threshold: float | None = None,
    threshold_source: str | None = None,
) -> dict[str, float | str | None]:
    """Evaluate frozen scores without fitting anything on the evaluated labels."""

    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    if y_true.shape != scores.shape:
        raise ValueError("y_true and scores must have identical shapes")
    pred = (scores >= threshold).astype(int) if threshold is not None else None
    p5, r5 = precision_recall_at_fraction(y_true, scores, 0.05)
    p10, r10 = precision_recall_at_fraction(y_true, scores, 0.10)
    return {
        "image_AUROC": _safe_metric(lambda: roc_auc_score(y_true, scores)),
        "image_AUPRC": _safe_metric(lambda: average_precision_score(y_true, scores)),
        "F1": _safe_metric(lambda: f1_score(y_true, pred, zero_division=0)) if pred is not None else None,
        "balanced_accuracy": _safe_metric(lambda: balanced_accuracy_score(y_true, pred)) if pred is not None else None,
        "Precision@5%": p5,
        "Precision@10%": p10,
        "Recall@5%": r5,
        "Recall@10%": r10,
        "threshold": threshold,
        "threshold_source": threshold_source or ("not_provided" if threshold is None else "unspecified_validation"),
    }


def pixel_anomaly_metrics(
    mask_flat: np.ndarray,
    score_flat: np.ndarray,
    *,
    threshold: float | None = None,
    threshold_source: str | None = None,
) -> dict[str, float | str | None]:
    mask_flat = np.asarray(mask_flat).astype(int)
    score_flat = np.asarray(score_flat).astype(float)
    out = {
        "pixel_AUROC": _safe_metric(lambda: roc_auc_score(mask_flat, score_flat)),
        "pixel_AUPRC": _safe_metric(lambda: average_precision_score(mask_flat, score_flat)),
        "PRO": None,
        "pixel_threshold": threshold,
        "pixel_threshold_source": threshold_source or ("not_provided" if threshold is None else "unspecified_validation"),
    }
    try:
        if threshold is None:
            return {**out, "IoU": None}
        pred = score_flat >= threshold
        inter = np.logical_and(pred, mask_flat == 1).sum()
        union = np.logical_or(pred, mask_flat == 1).sum()
        out["IoU"] = float(inter / max(union, 1))
    except Exception:
        out["IoU"] = None
    return out
