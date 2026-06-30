from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class DinoAvailability:
    available: bool
    model_name: str
    pretrained: bool
    message: str


class DinoBackbone(nn.Module):
    def __init__(self, model: nn.Module, model_name: str):
        super().__init__()
        self.model = model.eval()
        self.model_name = model_name
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward_features(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        out = self.model.forward_features(x)
        if isinstance(out, dict):
            tokens = out.get("x_norm_patchtokens")
            if tokens is None:
                tokens = out.get("patch_tokens")
        else:
            tokens = out
        if tokens is None:
            raise RuntimeError("DINO model did not return patch tokens")
        gh = x.shape[-2] // 14
        gw = x.shape[-1] // 14
        if tokens.ndim == 4:
            b, gh, gw, d = tokens.shape
            tokens = tokens.reshape(b, gh * gw, d)
        return F.normalize(tokens.float(), dim=-1), (gh, gw)


def build_dino_backbone(model_name: str = "dinov2_vits14", device: str = "cpu") -> tuple[DinoBackbone | None, DinoAvailability]:
    try:
        model = torch.hub.load("facebookresearch/dinov2", model_name, pretrained=True)
        backbone = DinoBackbone(model.to(device), model_name=model_name)
        return backbone, DinoAvailability(True, model_name, True, "loaded via torch.hub facebookresearch/dinov2")
    except Exception as exc:
        return None, DinoAvailability(False, model_name, False, f"unavailable: {type(exc).__name__}: {exc}")
