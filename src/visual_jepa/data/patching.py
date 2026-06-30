from __future__ import annotations

import torch


def random_block_mask_images(x: torch.Tensor, patch_size: int = 16, mask_ratio: float = 0.5) -> torch.Tensor:
    b, _, h, w = x.shape
    mask = torch.zeros((b, 1, h, w), dtype=torch.bool, device=x.device)
    gh, gw = max(1, h // patch_size), max(1, w // patch_size)
    n = max(1, int(round(gh * gw * mask_ratio)))
    for i in range(b):
        idx = torch.randperm(gh * gw, device=x.device)[:n]
        for flat in idx:
            yy = int(flat // gw)
            xx = int(flat % gw)
            mask[i, :, yy * patch_size : min((yy + 1) * patch_size, h), xx * patch_size : min((xx + 1) * patch_size, w)] = True
    return mask

