from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class TemporalMaskGenerator:
    num_tokens: int
    temporal_mask_ratio: float = 0.4
    num_target_spans: int = 2
    min_span: int = 1
    max_span: int = 4
    seed: int | None = None
    context_min_ratio: float = 0.25

    def _one(self, rng: np.random.Generator) -> dict[str, object]:
        if self.num_tokens < 2:
            raise ValueError("num_tokens must be at least 2")
        target = np.zeros(self.num_tokens, dtype=bool)
        desired = max(1, int(round(self.num_tokens * self.temporal_mask_ratio)))
        for _ in range(max(1, self.num_target_spans)):
            if int(target.sum()) >= desired:
                break
            span = int(rng.integers(self.min_span, max(self.min_span, self.max_span) + 1))
            span = min(span, self.num_tokens - 1)
            start = int(rng.integers(0, max(1, self.num_tokens - span + 1)))
            target[start : start + span] = True
        if int(target.sum()) == 0:
            target[int(rng.integers(0, self.num_tokens))] = True
        max_target = max(1, self.num_tokens - max(1, int(round(self.num_tokens * self.context_min_ratio))))
        if int(target.sum()) > max_target:
            idx = np.flatnonzero(target)
            rng.shuffle(idx)
            target[idx[max_target:]] = False
        context = ~target
        if int(context.sum()) == 0:
            target[-1] = False
            context = ~target
        blocks = _blocks_from_mask(target)
        return {
            "context_mask": torch.tensor(context, dtype=torch.bool),
            "target_mask": torch.tensor(target, dtype=torch.bool),
            "visible_mask": torch.tensor(context, dtype=torch.bool),
            "target_indices": torch.tensor(np.flatnonzero(target), dtype=torch.long),
            "target_blocks": blocks,
        }

    def __call__(self, batch_size: int | None = None, device: torch.device | str | None = None) -> dict[str, object]:
        rng = np.random.default_rng(self.seed)
        first = self._one(rng)
        if batch_size is None:
            if device is not None:
                for key in ["context_mask", "target_mask", "visible_mask", "target_indices"]:
                    first[key] = first[key].to(device)  # type: ignore[index, union-attr]
            return first
        context = first["context_mask"].unsqueeze(0).repeat(batch_size, 1)  # type: ignore[union-attr]
        target = first["target_mask"].unsqueeze(0).repeat(batch_size, 1)  # type: ignore[union-attr]
        visible = first["visible_mask"].unsqueeze(0).repeat(batch_size, 1)  # type: ignore[union-attr]
        indices = first["target_indices"].unsqueeze(0).repeat(batch_size, 1)  # type: ignore[union-attr]
        if device is not None:
            context = context.to(device)
            target = target.to(device)
            visible = visible.to(device)
            indices = indices.to(device)
        return {
            "context_mask": context,
            "target_mask": target,
            "visible_mask": visible,
            "target_indices": indices,
            "target_blocks": first["target_blocks"],
        }


def _blocks_from_mask(mask: np.ndarray) -> list[tuple[int, int]]:
    blocks = []
    start = None
    for i, value in enumerate(mask.tolist() + [False]):
        if value and start is None:
            start = i
        elif not value and start is not None:
            blocks.append((start, i))
            start = None
    return blocks
