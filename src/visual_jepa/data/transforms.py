from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image


def load_rgb(path: str | Path, image_size: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
    arr = np.asarray(img).astype("float32") / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr)


def load_mask(path: str | Path | None, image_size: int) -> torch.Tensor:
    if path is None or not Path(path).exists():
        return torch.zeros((1, image_size, image_size), dtype=torch.float32)
    img = Image.open(path).convert("L").resize((image_size, image_size), Image.Resampling.NEAREST)
    arr = (np.asarray(img) > 0).astype("float32")[None, ...]
    return torch.from_numpy(arr)


def normalize_image(x: torch.Tensor) -> torch.Tensor:
    return (x - 0.5) / 0.5


def denormalize_image(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x * 0.5 + 0.5, 0.0, 1.0)

