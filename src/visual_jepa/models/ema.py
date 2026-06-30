from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn


def clone_as_ema_target(module: nn.Module) -> nn.Module:
    target = deepcopy(module)
    for p in target.parameters():
        p.requires_grad_(False)
    target.eval()
    return target


@torch.no_grad()
def copy_weights(source: nn.Module, target: nn.Module) -> None:
    target.load_state_dict(source.state_dict())
    for p in target.parameters():
        p.requires_grad_(False)
    target.eval()


@torch.no_grad()
def update_ema(source: nn.Module, target: nn.Module, momentum: float = 0.996) -> None:
    momentum = float(momentum)
    source_state = source.state_dict()
    target_state = target.state_dict()
    for name, value in target_state.items():
        src = source_state[name].detach()
        if value.dtype.is_floating_point:
            value.mul_(momentum).add_(src.to(value.device, value.dtype), alpha=1.0 - momentum)
        else:
            value.copy_(src.to(value.device, value.dtype))
