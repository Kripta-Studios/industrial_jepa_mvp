from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def patch_scores_to_heatmap(patch_scores: torch.Tensor, grid_shape: tuple[int, int], output_size: tuple[int, int]) -> torch.Tensor:
    b = patch_scores.shape[0]
    heat = patch_scores.reshape(b, 1, grid_shape[0], grid_shape[1]).float()
    return F.interpolate(heat, size=output_size, mode="bilinear", align_corners=False).squeeze(1)


def save_heatmap_png(heatmap: torch.Tensor, path: str) -> None:
    arr = heatmap.detach().cpu().float().numpy()
    arr = (arr - arr.min()) / max(float(arr.max() - arr.min()), 1e-8)
    img = Image.fromarray((arr * 255).astype(np.uint8), mode="L")
    img.save(path)
