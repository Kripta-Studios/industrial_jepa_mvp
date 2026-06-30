from __future__ import annotations

import numpy as np


def nasa_score(y_true, y_pred) -> float:
    err = np.asarray(y_pred) - np.asarray(y_true)
    score = np.where(err < 0, np.exp(-err / 13) - 1, np.exp(err / 10) - 1)
    return float(np.sum(score))

