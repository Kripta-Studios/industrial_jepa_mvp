from __future__ import annotations

import numpy as np
import torch
from PIL import Image


def load_image_tensor(path: str, image_size: int = 224) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((image_size, image_size))
    arr = np.asarray(img).astype("float32") / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def normalize_image(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(3, 1, 1)
    return (x - mean) / std
