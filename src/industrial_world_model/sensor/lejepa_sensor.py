from __future__ import annotations

import numpy as np

from .features import engineered_sensor_features


def windows_to_engineered_matrix(windows: np.ndarray) -> np.ndarray:
    return np.stack([engineered_sensor_features(w) for w in windows])
