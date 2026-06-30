from __future__ import annotations

import torch

from .feature_memory import MemoryScores, PatchFeatureMemory


class PatchCoreLite(PatchFeatureMemory):
    """Small PatchCore-style scorer using a patch memory bank and nearest-neighbor distances."""

    def __init__(
        self,
        coreset_ratio: float = 0.1,
        coreset_method: str = "random",
        top_k: int = 5,
        seed: int = 42,
        max_memory_patches: int | None = 20000,
    ):
        super().__init__(
            coreset_ratio=coreset_ratio,
            coreset_method=coreset_method,
            top_k=top_k,
            seed=seed,
            max_memory_patches=max_memory_patches,
        )

    def fit_score(self, train_embeddings: torch.Tensor, test_embeddings: torch.Tensor) -> MemoryScores:
        self.fit(train_embeddings)
        return self.score(test_embeddings)
