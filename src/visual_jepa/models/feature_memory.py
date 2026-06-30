from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class MemoryScores:
    patch_scores: torch.Tensor
    image_scores: torch.Tensor


class PatchFeatureMemory:
    def __init__(
        self,
        coreset_ratio: float = 1.0,
        coreset_method: str = "random",
        top_k: int = 5,
        seed: int = 42,
        max_memory_patches: int | None = None,
    ):
        self.coreset_ratio = float(coreset_ratio)
        self.coreset_method = coreset_method
        self.top_k = int(top_k)
        self.seed = int(seed)
        self.max_memory_patches = max_memory_patches
        self.memory: torch.Tensor | None = None

    def fit(self, train_embeddings: torch.Tensor) -> "PatchFeatureMemory":
        if train_embeddings.ndim == 3:
            x = train_embeddings.reshape(-1, train_embeddings.shape[-1])
        elif train_embeddings.ndim == 2:
            x = train_embeddings
        else:
            raise ValueError("Expected train embeddings [B,N,D] or [M,D]")
        x = F.normalize(x.float(), dim=-1)
        n = len(x)
        keep = n
        if self.max_memory_patches is not None:
            keep = min(keep, int(self.max_memory_patches))
        if self.coreset_ratio < 1.0:
            keep = min(keep, max(1, int(round(n * self.coreset_ratio))))
        if keep < n:
            gen = torch.Generator(device=x.device).manual_seed(self.seed)
            if self.coreset_method == "random":
                idx = torch.randperm(n, generator=gen, device=x.device)[:keep]
            else:
                idx = greedy_farthest_subset(x, keep, seed=self.seed)
            x = x[idx]
        self.memory = x.contiguous()
        return self

    def score(self, embeddings: torch.Tensor, batch_patches: int = 4096) -> MemoryScores:
        if self.memory is None:
            raise RuntimeError("Memory bank is not fitted")
        if embeddings.ndim != 3:
            raise ValueError("Expected embeddings [B,N,D]")
        b, n, d = embeddings.shape
        flat = F.normalize(embeddings.float().reshape(-1, d), dim=-1)
        vals = []
        mem = self.memory.to(flat.device)
        for i in range(0, len(flat), batch_patches):
            dist = torch.cdist(flat[i : i + batch_patches], mem)
            vals.append(dist.min(dim=1).values)
        patch_scores = torch.cat(vals, dim=0).reshape(b, n)
        k = max(1, min(self.top_k, n))
        image_scores = patch_scores.topk(k, dim=1).values.mean(dim=1)
        return MemoryScores(patch_scores=patch_scores, image_scores=image_scores)


def greedy_farthest_subset(x: torch.Tensor, keep: int, seed: int = 42) -> torch.Tensor:
    n = len(x)
    gen = torch.Generator(device=x.device).manual_seed(seed)
    first = int(torch.randint(0, n, (1,), generator=gen, device=x.device).item())
    selected = [first]
    min_dist = torch.cdist(x[first : first + 1], x).squeeze(0)
    for _ in range(1, keep):
        idx = int(torch.argmax(min_dist).item())
        selected.append(idx)
        min_dist = torch.minimum(min_dist, torch.cdist(x[idx : idx + 1], x).squeeze(0))
    return torch.tensor(selected, dtype=torch.long, device=x.device)
