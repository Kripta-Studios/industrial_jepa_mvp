from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from .paths import ensure_dir


def safe_metric(fn, *args, default=None, **kwargs):
    try:
        return float(fn(*args, **kwargs))
    except Exception:
        return default


def threshold_from_validation(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    try:
        precision, recall, thresholds = precision_recall_curve(y_true, scores)
        if len(thresholds) == 0:
            return float(np.quantile(scores, 0.9))
        f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
        return float(thresholds[int(np.nanargmax(f1))])
    except Exception:
        return float(np.quantile(scores, 0.9))


def precision_recall_at_fraction(y_true: np.ndarray, scores: np.ndarray, fraction: float) -> tuple[float, float, float]:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    k = max(1, int(np.ceil(len(scores) * fraction)))
    order = np.argsort(-scores)
    pred = np.zeros_like(y_true)
    pred[order[:k]] = 1
    return (
        safe_metric(precision_score, y_true, pred, zero_division=0, default=0.0),
        safe_metric(recall_score, y_true, pred, zero_division=0, default=0.0),
        float(k),
    )


def expected_calibration_error(y_true: np.ndarray, scores: np.ndarray, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    scores = np.clip(scores, 0.0, 1.0)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (scores >= lo) & (scores < hi if hi < 1.0 else scores <= hi)
        if not np.any(mask):
            continue
        conf = float(np.mean(scores[mask]))
        acc = float(np.mean(y_true[mask]))
        ece += float(np.mean(mask)) * abs(acc - conf)
    return float(ece)


def false_alarms_per_tool(y_true: np.ndarray, pred: np.ndarray, tool_ids: np.ndarray | None) -> float | None:
    if tool_ids is None or len(tool_ids) == 0:
        return None
    df = pd.DataFrame({"tool": tool_ids, "y": y_true, "pred": pred})
    values = []
    for _, g in df.groupby("tool"):
        values.append(int(((g["pred"] == 1) & (g["y"] == 0)).sum()))
    return float(np.mean(values)) if values else None


def lead_time_stats(
    y_true: np.ndarray,
    pred: np.ndarray,
    tool_ids: np.ndarray | None,
    cycle_to_failure: np.ndarray | None,
) -> tuple[float | None, float | None]:
    if tool_ids is None or cycle_to_failure is None or len(tool_ids) == 0:
        return None, None
    df = pd.DataFrame({"tool": tool_ids, "y": y_true, "pred": pred, "ctf": cycle_to_failure})
    lead_times = []
    for _, g in df.groupby("tool"):
        hits = g[(g["pred"] == 1) & (g["y"] == 1)]
        if len(hits):
            lead_times.append(float(hits["ctf"].max()))
    if not lead_times:
        return None, None
    return float(np.mean(lead_times)), float(np.median(lead_times))


def forecast_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    tool_ids: np.ndarray | None = None,
    cycle_to_failure: np.ndarray | None = None,
    prefix: str = "",
) -> dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    pred = (scores >= threshold).astype(int)
    p5, r5, k5 = precision_recall_at_fraction(y_true, scores, 0.05)
    p10, r10, k10 = precision_recall_at_fraction(y_true, scores, 0.10)
    mean_lead, median_lead = lead_time_stats(y_true, pred, tool_ids, cycle_to_failure)
    key = lambda name: f"{prefix}{name}" if prefix else name
    out = {
        key("AUROC"): safe_metric(roc_auc_score, y_true, scores),
        key("AUPRC"): safe_metric(average_precision_score, y_true, scores),
        key("precision_at_5pct"): p5,
        key("recall_at_5pct"): r5,
        key("alerts_at_5pct"): k5,
        key("precision_at_10pct"): p10,
        key("recall_at_10pct"): r10,
        key("alerts_at_10pct"): k10,
        key("F1"): safe_metric(f1_score, y_true, pred, zero_division=0, default=0.0),
        key("balanced_accuracy"): safe_metric(balanced_accuracy_score, y_true, pred, default=None),
        key("brier_score"): safe_metric(brier_score_loss, y_true, np.clip(scores, 0.0, 1.0), default=None),
        key("ECE"): expected_calibration_error(y_true, scores),
        key("threshold"): float(threshold),
        key("false_alarms_per_tool"): false_alarms_per_tool(y_true, pred, tool_ids),
        key("mean_lead_time"): mean_lead,
        key("median_lead_time"): median_lead,
    }
    return out


def metrics_by_tool(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    tool_ids: np.ndarray,
    prefix: str = "",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    df = pd.DataFrame({"tool": tool_ids, "y": y_true, "score": scores})
    for tool, g in df.groupby("tool"):
        pred = (g["score"].to_numpy() >= threshold).astype(int)
        row = {
            "tool_id": tool,
            "n": len(g),
            "failure_rate": float(g["y"].mean()),
            "false_alarms": int(((pred == 1) & (g["y"].to_numpy() == 0)).sum()),
        }
        row.update(forecast_metrics(g["y"].to_numpy(), g["score"].to_numpy(), threshold, prefix=prefix))
        rows.append(row)
    return rows


def save_forecast_plots(
    y_true: np.ndarray,
    scores: np.ndarray,
    out_dir: str | Path,
    title_prefix: str,
    tool_ids: np.ndarray | None = None,
    cycle_ids: np.ndarray | None = None,
    threshold: float | None = None,
    cycle_to_failure: np.ndarray | None = None,
) -> dict[str, str]:
    out_dir = ensure_dir(out_dir)
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    paths: dict[str, str] = {}

    if len(np.unique(y_true)) > 1:
        fpr, tpr, _ = roc_curve(y_true, scores)
        plt.figure(figsize=(5, 4))
        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.title(f"{title_prefix} ROC")
        plt.tight_layout()
        p = out_dir / "roc_curve.png"
        plt.savefig(p, dpi=150)
        plt.close()
        paths["roc_curve"] = str(p)

        precision, recall, _ = precision_recall_curve(y_true, scores)
        plt.figure(figsize=(5, 4))
        plt.plot(recall, precision)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"{title_prefix} PR")
        plt.tight_layout()
        p = out_dir / "precision_recall_curve.png"
        plt.savefig(p, dpi=150)
        plt.close()
        paths["precision_recall_curve"] = str(p)

    plt.figure(figsize=(5, 4))
    plt.hist(scores[y_true == 0], bins=20, alpha=0.7, label="negative")
    plt.hist(scores[y_true == 1], bins=20, alpha=0.7, label="positive")
    plt.xlabel("Risk score")
    plt.ylabel("Count")
    plt.title(f"{title_prefix} score distribution")
    plt.legend()
    plt.tight_layout()
    p = out_dir / "score_distribution.png"
    plt.savefig(p, dpi=150)
    plt.close()
    paths["score_distribution"] = str(p)

    bins = np.linspace(0, 1, 11)
    bin_centers, observed = [], []
    clipped = np.clip(scores, 0, 1)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (clipped >= lo) & (clipped < hi if hi < 1 else clipped <= hi)
        if np.any(mask):
            bin_centers.append(float((lo + hi) / 2))
            observed.append(float(np.mean(y_true[mask])))
    plt.figure(figsize=(5, 4))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    if bin_centers:
        plt.plot(bin_centers, observed, marker="o")
    plt.xlabel("Predicted probability/bin")
    plt.ylabel("Observed failure rate")
    plt.title(f"{title_prefix} calibration")
    plt.tight_layout()
    p = out_dir / "calibration_curve.png"
    plt.savefig(p, dpi=150)
    plt.close()
    paths["calibration_curve"] = str(p)

    if tool_ids is not None and cycle_ids is not None and len(tool_ids):
        df = pd.DataFrame({"tool": tool_ids, "cycle": cycle_ids, "score": scores, "label": y_true})
        for tool, g in list(df.groupby("tool"))[:6]:
            g = g.sort_values("cycle")
            plt.figure(figsize=(7, 3))
            plt.plot(g["cycle"], g["score"], label="risk")
            positives = g[g["label"] == 1]
            if len(positives):
                plt.scatter(positives["cycle"], positives["score"], color="red", label="failure soon")
            plt.xlabel("Cycle")
            plt.ylabel("Risk score")
            plt.title(f"{title_prefix} risk timeline tool {tool}")
            plt.legend()
            plt.tight_layout()
            p = out_dir / f"risk_timeline_tool_{tool}.png"
            plt.savefig(p, dpi=150)
            plt.close()
            paths[f"risk_timeline_tool_{tool}"] = str(p)
    if threshold is not None and cycle_to_failure is not None and len(cycle_to_failure):
        pred = scores >= threshold
        lead_times = np.asarray(cycle_to_failure)[(pred == 1) & (y_true == 1)]
        if len(lead_times):
            plt.figure(figsize=(5, 4))
            plt.hist(lead_times, bins=min(10, len(np.unique(lead_times))))
            plt.xlabel("CycleToFailure at alert")
            plt.ylabel("Count")
            plt.title(f"{title_prefix} lead-time histogram")
            plt.tight_layout()
            p = out_dir / "lead_time_histogram.png"
            plt.savefig(p, dpi=150)
            plt.close()
            paths["lead_time_histogram"] = str(p)
    return paths
