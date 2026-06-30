from __future__ import annotations

import torch


def rank_actions_by_predicted_surprise(candidate_actions: torch.Tensor, z_t: torch.Tensor, model, reference_z: torch.Tensor) -> torch.Tensor:
    preds = model(z_t.repeat(candidate_actions.shape[0], 1), candidate_actions)
    scores = (preds - reference_z.repeat(candidate_actions.shape[0], 1)).pow(2).mean(dim=1)
    return torch.argsort(scores)
