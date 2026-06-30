from __future__ import annotations

import math

import torch
from torch import nn

from sensor_jepa.data.temporal_tokenization import TemporalPatchTokenizer


class TemporalTransformerEncoder(nn.Module):
    def __init__(
        self,
        input_channels: int,
        embedding_dim: int = 256,
        depth: int = 4,
        num_heads: int = 4,
        temporal_patch_size: int = 2,
        temporal_patch_stride: int = 1,
        tokenization_mode: str = "multichannel_token",
        max_tokens: int = 512,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.tokenizer = TemporalPatchTokenizer(
            input_channels=input_channels,
            embedding_dim=embedding_dim,
            temporal_patch_size=temporal_patch_size,
            temporal_patch_stride=temporal_patch_stride,
            mode=tokenization_mode,
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, max_tokens, embedding_dim))
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=embedding_dim,
                    nhead=num_heads,
                    dim_feedforward=embedding_dim * 4,
                    dropout=0.0,
                    batch_first=True,
                    activation="gelu",
                    norm_first=True,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embedding_dim)
        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor, return_hidden_states: bool = False) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        tokens = self.tokenizer(x)
        n = tokens.shape[1]
        if n > self.pos_embed.shape[1]:
            raise ValueError(f"num tokens {n} exceeds max_tokens {self.pos_embed.shape[1]}")
        h = tokens * math.sqrt(self.embedding_dim) + self.pos_embed[:, :n]
        hidden = []
        for layer in self.layers:
            h = layer(h)
            if return_hidden_states:
                hidden.append(self.norm(h))
        h = self.norm(h)
        out: dict[str, torch.Tensor | list[torch.Tensor]] = {"tokens": h}
        if return_hidden_states:
            out["hidden_states"] = hidden
        return out

    def positional_embeddings(self, indices: torch.Tensor) -> torch.Tensor:
        if indices.ndim == 1:
            return self.pos_embed[:, indices]
        return self.pos_embed[0, indices]


class TemporalConvTokenEncoder(nn.Module):
    def __init__(
        self,
        input_channels: int,
        embedding_dim: int = 256,
        hidden_dim: int = 256,
        temporal_patch_size: int = 2,
        temporal_patch_stride: int = 1,
        max_tokens: int = 512,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.conv = nn.Sequential(
            nn.Conv1d(input_channels, hidden_dim, kernel_size=temporal_patch_size, stride=temporal_patch_stride),
            nn.GELU(),
            nn.Conv1d(hidden_dim, embedding_dim, kernel_size=1),
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, max_tokens, embedding_dim))
        self.norm = nn.LayerNorm(embedding_dim)
        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor, return_hidden_states: bool = False) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        if x.shape[1] != self.conv[0].in_channels and x.shape[-1] == self.conv[0].in_channels:
            x = x.transpose(1, 2)
        h = self.conv(x).transpose(1, 2)
        n = h.shape[1]
        if n > self.pos_embed.shape[1]:
            raise ValueError(f"num tokens {n} exceeds max_tokens {self.pos_embed.shape[1]}")
        h = self.norm(h + self.pos_embed[:, :n])
        out: dict[str, torch.Tensor | list[torch.Tensor]] = {"tokens": h}
        if return_hidden_states:
            out["hidden_states"] = [h]
        return out

    def positional_embeddings(self, indices: torch.Tensor) -> torch.Tensor:
        if indices.ndim == 1:
            return self.pos_embed[:, indices]
        return self.pos_embed[0, indices]


def build_dense_sensor_encoder(
    kind: str,
    input_channels: int,
    embedding_dim: int,
    depth: int,
    num_heads: int,
    temporal_patch_size: int,
    temporal_patch_stride: int,
    tokenization_mode: str = "multichannel_token",
) -> nn.Module:
    if kind == "temporal_transformer":
        return TemporalTransformerEncoder(
            input_channels=input_channels,
            embedding_dim=embedding_dim,
            depth=depth,
            num_heads=num_heads,
            temporal_patch_size=temporal_patch_size,
            temporal_patch_stride=temporal_patch_stride,
            tokenization_mode=tokenization_mode,
        )
    if kind == "temporal_conv":
        return TemporalConvTokenEncoder(
            input_channels=input_channels,
            embedding_dim=embedding_dim,
            hidden_dim=embedding_dim,
            temporal_patch_size=temporal_patch_size,
            temporal_patch_stride=temporal_patch_stride,
        )
    raise ValueError(f"Unknown dense sensor encoder: {kind}")
