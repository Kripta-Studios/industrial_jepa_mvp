from __future__ import annotations

import torch


class PixelStatBaseline:
    def __init__(self):
        self.mean: torch.Tensor | None = None
        self.std: torch.Tensor | None = None

    def fit(self, images: torch.Tensor) -> "PixelStatBaseline":
        x = images.float()
        self.mean = x.mean(dim=(0, 2, 3), keepdim=True)
        self.std = x.std(dim=(0, 2, 3), keepdim=True).clamp_min(1e-6)
        return self

    def score(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.mean is None or self.std is None:
            raise RuntimeError("PixelStatBaseline is not fitted")
        z = ((images.float() - self.mean.to(images.device)) / self.std.to(images.device)).abs()
        heatmap = z.mean(dim=1)
        score = heatmap.flatten(1).topk(max(1, heatmap.shape[-1] * heatmap.shape[-2] // 20), dim=1).values.mean(dim=1)
        return score, heatmap
