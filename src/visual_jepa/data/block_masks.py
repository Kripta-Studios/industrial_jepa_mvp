from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class BlockMaskBatch:
    context_mask: torch.Tensor
    target_mask: torch.Tensor
    visible_mask: torch.Tensor
    target_blocks: list[list[dict[str, int]]]
    target_indices: torch.Tensor
    target_valid: torch.Tensor
    visible_indices: torch.Tensor
    visible_valid: torch.Tensor


def _pad_indices(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rows = []
    valid = []
    max_len = max(int(m.sum().item()) for m in mask)
    max_len = max(max_len, 1)
    for m in mask:
        idx = torch.nonzero(m, as_tuple=False).flatten()
        row = torch.zeros(max_len, dtype=torch.long, device=mask.device)
        val = torch.zeros(max_len, dtype=torch.bool, device=mask.device)
        if len(idx):
            row[: len(idx)] = idx
            val[: len(idx)] = True
        rows.append(row)
        valid.append(val)
    return torch.stack(rows, dim=0), torch.stack(valid, dim=0)


class BlockMaskGenerator:
    def __init__(
        self,
        mask_ratio: float = 0.6,
        num_target_blocks: int = 4,
        min_block_size: int = 2,
        max_block_size: int = 6,
        context_min_ratio: float = 0.25,
        seed: int | None = None,
    ):
        self.mask_ratio = float(mask_ratio)
        self.num_target_blocks = int(num_target_blocks)
        self.min_block_size = int(min_block_size)
        self.max_block_size = int(max_block_size)
        self.context_min_ratio = float(context_min_ratio)
        self.seed = seed
        self._calls = 0

    def _generator(self, device: torch.device | str) -> torch.Generator:
        gen = torch.Generator(device=device)
        if self.seed is not None:
            gen.manual_seed(int(self.seed) + self._calls)
        else:
            gen.seed()
        self._calls += 1
        return gen

    def __call__(self, batch_size: int, grid_shape: tuple[int, int], device: torch.device | str = "cpu") -> BlockMaskBatch:
        gh, gw = grid_shape
        n = gh * gw
        target = torch.zeros((batch_size, n), dtype=torch.bool, device=device)
        blocks: list[list[dict[str, int]]] = []
        gen = self._generator(device)
        target_count = max(1, int(round(n * self.mask_ratio)))
        min_visible = max(1, int(round(n * self.context_min_ratio)))
        max_target = max(1, n - min_visible)
        target_count = min(target_count, max_target)

        for b in range(batch_size):
            sample_blocks: list[dict[str, int]] = []
            attempts = 0
            while int(target[b].sum().item()) < target_count and attempts < self.num_target_blocks * 16:
                attempts += 1
                max_h = max(1, min(self.max_block_size, gh))
                max_w = max(1, min(self.max_block_size, gw))
                min_h = max(1, min(self.min_block_size, max_h))
                min_w = max(1, min(self.min_block_size, max_w))
                bh = int(torch.randint(min_h, max_h + 1, (1,), generator=gen, device=device).item())
                bw = int(torch.randint(min_w, max_w + 1, (1,), generator=gen, device=device).item())
                y0 = int(torch.randint(0, gh - bh + 1, (1,), generator=gen, device=device).item())
                x0 = int(torch.randint(0, gw - bw + 1, (1,), generator=gen, device=device).item())
                sample_blocks.append({"y0": y0, "x0": x0, "h": bh, "w": bw})
                yy = torch.arange(y0, y0 + bh, device=device)
                xx = torch.arange(x0, x0 + bw, device=device)
                grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
                idx = (grid_y * gw + grid_x).flatten()
                target[b, idx] = True
                if len(sample_blocks) >= self.num_target_blocks and int(target[b].sum().item()) >= target_count:
                    break

            if int(target[b].sum().item()) > target_count:
                idx = torch.nonzero(target[b], as_tuple=False).flatten()
                keep = idx[torch.randperm(len(idx), generator=gen, device=device)[:target_count]]
                target[b].zero_()
                target[b, keep] = True
            if int(target[b].sum().item()) == 0:
                target[b, int(torch.randint(0, n, (1,), generator=gen, device=device).item())] = True
            if int((~target[b]).sum().item()) == 0:
                target[b, 0] = False
            blocks.append(sample_blocks)

        visible = ~target
        context = visible.clone()
        target_idx, target_valid = _pad_indices(target)
        visible_idx, visible_valid = _pad_indices(visible)
        return BlockMaskBatch(
            context_mask=context,
            target_mask=target,
            visible_mask=visible,
            target_blocks=blocks,
            target_indices=target_idx,
            target_valid=target_valid,
            visible_indices=visible_idx,
            visible_valid=visible_valid,
        )
