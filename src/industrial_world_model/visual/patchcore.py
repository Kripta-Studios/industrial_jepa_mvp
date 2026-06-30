from __future__ import annotations

import torch

try:
    from visual_jepa.models.patchcore_lite import PatchCoreLite as _PatchCoreLite
except Exception:  # pragma: no cover - fallback for isolated package use
    _PatchCoreLite = None


class PatchCoreLite:
    def __init__(self, coreset_ratio: float = 0.1, top_k: int = 5, seed: int = 42):
        if _PatchCoreLite is None:
            self._impl = None
            self.memory = None
            self.top_k = top_k
        else:
            self._impl = _PatchCoreLite(coreset_ratio=coreset_ratio, top_k=top_k, seed=seed)

    def fit(self, train_embeddings: torch.Tensor) -> "PatchCoreLite":
        if self._impl is not None:
            self._impl.fit(train_embeddings)
        else:
            self.memory = train_embeddings.reshape(-1, train_embeddings.shape[-1]).float()
        return self

    def score(self, embeddings: torch.Tensor):
        if self._impl is not None:
            return self._impl.score(embeddings)
        x = embeddings.float()
        dist = torch.cdist(x.reshape(-1, x.shape[-1]), self.memory)
        patch_scores = dist.min(dim=1).values.reshape(x.shape[0], x.shape[1])
        image_scores = patch_scores.topk(min(self.top_k, x.shape[1]), dim=1).values.mean(dim=1)
        return type("PatchCoreScores", (), {"patch_scores": patch_scores, "image_scores": image_scores})

    def fit_score(self, train_embeddings: torch.Tensor, test_embeddings: torch.Tensor):
        return self.fit(train_embeddings).score(test_embeddings)
