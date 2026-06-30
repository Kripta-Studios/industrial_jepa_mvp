from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .paths import ensure_dir


def plot_history(history: list[dict], path: str | Path, y_key: str = "loss") -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    xs = [h.get("epoch", i + 1) for i, h in enumerate(history)]
    ys = [h.get(y_key) for h in history]
    plt.figure(figsize=(6, 4))
    plt.plot(xs, ys, marker="o")
    plt.xlabel("epoch")
    plt.ylabel(y_key)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def save_heatmap_overlay(
    image: np.ndarray,
    heatmap: np.ndarray,
    mask: np.ndarray | None,
    path: str | Path,
    title: str = "",
) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[0] in (1, 3):
        image = np.transpose(image, (1, 2, 0))
    image = np.clip(image, 0, 1)
    heatmap = np.asarray(heatmap)
    hmin, hmax = float(np.nanmin(heatmap)), float(np.nanmax(heatmap))
    heatmap = (heatmap - hmin) / max(hmax - hmin, 1e-8)
    ncols = 3 if mask is not None else 2
    plt.figure(figsize=(4 * ncols, 4))
    plt.subplot(1, ncols, 1)
    plt.imshow(image)
    plt.axis("off")
    plt.title("input")
    plt.subplot(1, ncols, 2)
    plt.imshow(image)
    plt.imshow(heatmap, cmap="magma", alpha=0.55)
    plt.axis("off")
    plt.title(title or "anomaly heatmap")
    if mask is not None:
        plt.subplot(1, ncols, 3)
        plt.imshow(mask, cmap="gray")
        plt.axis("off")
        plt.title("gt mask")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path

