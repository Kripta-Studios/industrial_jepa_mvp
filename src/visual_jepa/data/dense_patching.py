from __future__ import annotations

import torch
import torch.nn.functional as F


def pad_to_patch_size(x: torch.Tensor, patch_size: int) -> tuple[torch.Tensor, tuple[int, int]]:
    """Pad an image batch so height and width are divisible by patch_size."""
    if x.ndim != 4:
        raise ValueError(f"Expected [B,C,H,W], got {tuple(x.shape)}")
    _, _, h, w = x.shape
    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h))
    return x, (pad_h, pad_w)


def patchify(x: torch.Tensor, patch_size: int) -> tuple[torch.Tensor, tuple[int, int]]:
    """Convert images [B,C,H,W] to flattened patches [B,N,C*P*P]."""
    x, _ = pad_to_patch_size(x, patch_size)
    b, c, h, w = x.shape
    gh, gw = h // patch_size, w // patch_size
    patches = x.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(b, gh * gw, c * patch_size * patch_size)
    return patches, (gh, gw)


def unpatchify(patches: torch.Tensor, grid_shape: tuple[int, int], patch_size: int, channels: int) -> torch.Tensor:
    """Convert flattened patches [B,N,C*P*P] back to images [B,C,H,W]."""
    if patches.ndim != 3:
        raise ValueError(f"Expected [B,N,D], got {tuple(patches.shape)}")
    gh, gw = grid_shape
    b, n, dim = patches.shape
    expected = gh * gw
    if n != expected:
        raise ValueError(f"Patch count {n} does not match grid {grid_shape}")
    if dim != channels * patch_size * patch_size:
        raise ValueError("Patch dimension does not match channels and patch_size")
    x = patches.reshape(b, gh, gw, channels, patch_size, patch_size)
    x = x.permute(0, 3, 1, 4, 2, 5).reshape(b, channels, gh * patch_size, gw * patch_size)
    return x


def patch_mask_to_image(mask: torch.Tensor, grid_shape: tuple[int, int], patch_size: int) -> torch.Tensor:
    """Upsample a patch mask [B,N] or [B,1,gh,gw] to image mask [B,1,H,W]."""
    gh, gw = grid_shape
    if mask.ndim == 2:
        mask = mask.reshape(mask.shape[0], 1, gh, gw)
    if mask.ndim != 4:
        raise ValueError(f"Expected [B,N] or [B,1,gh,gw], got {tuple(mask.shape)}")
    return F.interpolate(mask.float(), size=(gh * patch_size, gw * patch_size), mode="nearest")


def grid_positions(grid_shape: tuple[int, int], device: torch.device | str | None = None) -> torch.Tensor:
    """Return normalized [N,2] patch-grid coordinates in [-1,1]."""
    gh, gw = grid_shape
    ys = torch.linspace(-1.0, 1.0, gh, device=device)
    xs = torch.linspace(-1.0, 1.0, gw, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([yy, xx], dim=-1).reshape(gh * gw, 2)
