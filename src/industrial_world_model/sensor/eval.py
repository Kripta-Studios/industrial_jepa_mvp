from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def sensor_scores_report(y: np.ndarray, scores: np.ndarray) -> dict[str, float | None]:
    try:
        auroc = float(roc_auc_score(y, scores))
    except Exception:
        auroc = None
    try:
        auprc = float(average_precision_score(y, scores))
    except Exception:
        auprc = None
    return {"AUROC": auroc, "AUPRC": auprc}
