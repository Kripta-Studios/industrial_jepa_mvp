from __future__ import annotations

import torch
from torch import nn


def count_temporal_tokens(length: int, patch_size: int, stride: int) -> int:
    if length < patch_size:
        return 0
    return 1 + (length - patch_size) // stride


def temporal_patchify(
    x: torch.Tensor,
    patch_size: int,
    stride: int,
    mode: str = "multichannel_token",
) -> torch.Tensor:
    """Patchify [B, T, C] windows into temporal tokens."""

    if x.ndim != 3:
        raise ValueError(f"Expected [B, T, C], got shape {tuple(x.shape)}")
    if x.shape[1] < patch_size:
        raise ValueError(f"temporal length {x.shape[1]} is smaller than patch_size {patch_size}")
    patches = x.unfold(dimension=1, size=patch_size, step=stride)  # [B, N, C, P]
    if mode == "multichannel_token":
        return patches.permute(0, 1, 3, 2).contiguous().flatten(start_dim=2)
    if mode == "channel_token":
        b, n, c, p = patches.shape
        return patches.permute(0, 1, 2, 3).contiguous().view(b, n * c, p)
    raise ValueError(f"Unknown temporal tokenization mode: {mode}")


class TemporalPatchTokenizer(nn.Module):
    def __init__(
        self,
        input_channels: int,
        embedding_dim: int,
        temporal_patch_size: int,
        temporal_patch_stride: int,
        mode: str = "multichannel_token",
    ):
        super().__init__()
        self.input_channels = input_channels
        self.embedding_dim = embedding_dim
        self.temporal_patch_size = temporal_patch_size
        self.temporal_patch_stride = temporal_patch_stride
        self.mode = mode
        input_dim = temporal_patch_size * input_channels if mode == "multichannel_token" else temporal_patch_size
        self.proj = nn.Linear(input_dim, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.input_channels and x.shape[1] == self.input_channels:
            x = x.transpose(1, 2)
        patches = temporal_patchify(
            x,
            patch_size=self.temporal_patch_size,
            stride=self.temporal_patch_stride,
            mode=self.mode,
        )
        return self.proj(patches)
