from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score


def frozen_logistic_probe(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray) -> dict[str, float | None]:
    try:
        clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(x_train, y_train)
        score = clf.predict_proba(x_test)[:, 1]
        return {"probe_AUROC": float(roc_auc_score(y_test, score)), "probe_AUPRC": float(average_precision_score(y_test, score))}
    except Exception:
        return {"probe_AUROC": None, "probe_AUPRC": None}
