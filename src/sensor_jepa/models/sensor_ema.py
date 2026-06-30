from __future__ import annotations

import copy

import torch
from torch import nn


def clone_as_ema_target(module: nn.Module) -> nn.Module:
    target = copy.deepcopy(module)
    for param in target.parameters():
        param.requires_grad_(False)
    target.eval()
    return target


@torch.no_grad()
def copy_weights(source: nn.Module, target: nn.Module) -> None:
    target.load_state_dict(source.state_dict())
    for param in target.parameters():
        param.requires_grad_(False)


@torch.no_grad()
def update_ema(source: nn.Module, target: nn.Module, momentum: float = 0.996) -> None:
    if not 0.0 <= momentum <= 1.0:
        raise ValueError("EMA momentum must be in [0, 1].")
    source_state = source.state_dict()
    target_state = target.state_dict()
    for name, target_value in target_state.items():
        source_value = source_state[name].detach()
        if torch.is_floating_point(target_value):
            target_value.mul_(momentum).add_(source_value, alpha=1.0 - momentum)
        else:
            target_value.copy_(source_value)
