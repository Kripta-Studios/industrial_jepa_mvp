from __future__ import annotations

import numpy as np


def sliding_windows(
    values: np.ndarray,
    window_length: int,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Return sliding windows and their last-row indices."""
    values = np.asarray(values)
    if values.ndim != 2:
        raise ValueError(f"values must be [N, C], got {values.shape}")
    if window_length <= 0:
        raise ValueError("window_length must be positive")
    if len(values) < window_length:
        return np.empty((0, window_length, values.shape[1]), dtype=values.dtype), np.empty((0,), dtype=int)
    starts = np.arange(0, len(values) - window_length + 1, stride)
    windows = np.stack([values[i : i + window_length] for i in starts], axis=0)
    last_indices = starts + window_length - 1
    return windows, last_indices


def stratified_label_fraction_indices(
    y: np.ndarray,
    fraction: float,
    seed: int = 42,
) -> np.ndarray:
    y = np.asarray(y)
    fraction = float(fraction)
    if fraction >= 1:
        return np.arange(len(y))
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        n = max(1, int(round(len(idx) * fraction)))
        selected.extend(rng.choice(idx, size=min(n, len(idx)), replace=False).tolist())
    selected = np.array(sorted(selected), dtype=int)
    return selected

