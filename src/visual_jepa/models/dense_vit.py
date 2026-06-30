from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from visual_jepa.data.dense_patching import grid_positions, pad_to_patch_size


@dataclass
class DenseEncoderOutput:
    tokens: torch.Tensor
    hidden_states: list[torch.Tensor]
    grid_shape: tuple[int, int]
    positions: torch.Tensor


class DenseViTEncoder(nn.Module):
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embedding_dim: int = 192,
        depth: int = 6,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        intermediate_layers: tuple[int, ...] | None = None,
    ):
        super().__init__()
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.embedding_dim = int(embedding_dim)
        self.depth = int(depth)
        self.patch_embed = nn.Conv2d(in_channels, embedding_dim, kernel_size=patch_size, stride=patch_size)
        self.base_grid = (max(1, image_size // patch_size), max(1, image_size // patch_size))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.base_grid[0] * self.base_grid[1], embedding_dim))
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=embedding_dim,
                    nhead=num_heads,
                    dim_feedforward=int(embedding_dim * mlp_ratio),
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embedding_dim)
        if intermediate_layers is None:
            if depth >= 4:
                intermediate_layers = (depth // 4, depth // 2, (3 * depth) // 4, depth)
            else:
                intermediate_layers = tuple(range(1, depth + 1))
        self.intermediate_layers = tuple(sorted(set(int(i) for i in intermediate_layers if 1 <= int(i) <= depth)))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def _pos_embed(self, grid_shape: tuple[int, int]) -> torch.Tensor:
        gh, gw = grid_shape
        if grid_shape == self.base_grid:
            return self.pos_embed
        base_h, base_w = self.base_grid
        pos = self.pos_embed.reshape(1, base_h, base_w, self.embedding_dim).permute(0, 3, 1, 2)
        pos = F.interpolate(pos, size=(gh, gw), mode="bicubic", align_corners=False)
        return pos.permute(0, 2, 3, 1).reshape(1, gh * gw, self.embedding_dim)

    def forward_tokens(self, x: torch.Tensor) -> DenseEncoderOutput:
        x, _ = pad_to_patch_size(x, self.patch_size)
        patch = self.patch_embed(x)
        b, d, gh, gw = patch.shape
        tokens = patch.flatten(2).transpose(1, 2)
        tokens = tokens + self._pos_embed((gh, gw)).to(tokens.device, tokens.dtype)
        hidden: list[torch.Tensor] = []
        for i, block in enumerate(self.blocks, start=1):
            tokens = block(tokens)
            if i in self.intermediate_layers:
                hidden.append(self.norm(tokens))
        tokens = self.norm(tokens)
        if not hidden or hidden[-1] is not tokens:
            if self.depth not in self.intermediate_layers:
                hidden.append(tokens)
        positions = grid_positions((gh, gw), device=tokens.device).to(tokens.dtype)
        return DenseEncoderOutput(tokens=tokens, hidden_states=hidden, grid_shape=(gh, gw), positions=positions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_tokens(x).tokens


def build_dense_vit_encoder(
    name: str = "vit_tiny",
    image_size: int = 224,
    patch_size: int = 16,
    in_channels: int = 3,
    embedding_dim: int | None = None,
    depth: int | None = None,
    num_heads: int | None = None,
) -> DenseViTEncoder:
    presets = {
        "vit_tiny": {"embedding_dim": 192, "depth": 6, "num_heads": 3},
        "vit_small": {"embedding_dim": 384, "depth": 8, "num_heads": 6},
    }
    if name not in presets:
        raise ValueError(f"Unknown dense ViT encoder: {name}")
    p = presets[name].copy()
    if embedding_dim is not None:
        p["embedding_dim"] = int(embedding_dim)
    if depth is not None:
        p["depth"] = int(depth)
    if num_heads is not None:
        p["num_heads"] = int(num_heads)
    return DenseViTEncoder(
        image_size=image_size,
        patch_size=patch_size,
        in_channels=in_channels,
        embedding_dim=p["embedding_dim"],
        depth=p["depth"],
        num_heads=p["num_heads"],
    )
