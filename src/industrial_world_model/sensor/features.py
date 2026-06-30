from __future__ import annotations

import numpy as np


def engineered_sensor_features(window: np.ndarray) -> np.ndarray:
    x = np.asarray(window, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    rms = np.sqrt((x**2).mean(axis=0))
    peak_to_peak = x.max(axis=0) - x.min(axis=0)
    energy = (x**2).sum(axis=0)
    zcr = ((x[:-1] * x[1:]) < 0).mean(axis=0) if x.shape[0] > 1 else np.zeros(x.shape[1])
    return np.concatenate([mean, std, rms, peak_to_peak, energy, zcr]).astype("float32")
