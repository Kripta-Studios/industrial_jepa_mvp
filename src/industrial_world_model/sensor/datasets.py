from __future__ import annotations

from pathlib import Path

import numpy as np


def load_or_synthetic_sensor_windows(root: str | Path, max_windows: int = 128, window_size: int = 128, channels: int = 3) -> tuple[np.ndarray, np.ndarray]:
    root = Path(root)
    csvs = list(root.rglob("*.csv")) if root.exists() else []
    if csvs:
        try:
            import pandas as pd

            df = pd.read_csv(csvs[0]).select_dtypes("number").fillna(0)
            arr = df.to_numpy(dtype="float32")
            if arr.ndim == 1:
                arr = arr[:, None]
            windows = []
            for start in range(0, max(1, len(arr) - window_size), window_size):
                windows.append(arr[start : start + window_size, :channels])
                if len(windows) >= max_windows:
                    break
            x = np.stack(windows) if windows else np.random.randn(max_windows, window_size, channels).astype("float32")
            y = np.zeros(len(x), dtype=int)
            return x, y
        except Exception:
            pass
    rng = np.random.default_rng(42)
    x = rng.normal(size=(max_windows, window_size, channels)).astype("float32")
    y = np.zeros(max_windows, dtype=int)
    y[-max(1, max_windows // 5) :] = 1
    x[y == 1] += 0.8
    return x, y
