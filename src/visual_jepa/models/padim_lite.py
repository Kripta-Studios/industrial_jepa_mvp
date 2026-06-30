from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class PadimScores:
    patch_scores: torch.Tensor
    image_scores: torch.Tensor


class PadimLite:
    def __init__(self, n_features: int | None = 100, eps: float = 1e-3, seed: int = 42, top_k: int = 5):
        self.n_features = n_features
        self.eps = float(eps)
        self.seed = int(seed)
        self.top_k = int(top_k)
        self.feature_idx: torch.Tensor | None = None
        self.mean: torch.Tensor | None = None
        self.inv_cov: torch.Tensor | None = None

    def _select(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1]
        if self.feature_idx is None:
            if self.n_features is None or self.n_features >= d:
                self.feature_idx = torch.arange(d, device=x.device)
            else:
                gen = torch.Generator(device=x.device).manual_seed(self.seed)
                self.feature_idx = torch.randperm(d, generator=gen, device=x.device)[: int(self.n_features)]
        return x[..., self.feature_idx.to(x.device)]

    def fit(self, train_embeddings: torch.Tensor) -> "PadimLite":
        if train_embeddings.ndim != 3:
            raise ValueError("Expected train embeddings [B,N,D]")
        x = F.normalize(train_embeddings.float(), dim=-1)
        x = self._select(x)
        b, n, d = x.shape
        mean = x.mean(dim=0)
        centered = x - mean.unsqueeze(0)
        cov = torch.einsum("bnd,bne->nde", centered, centered) / max(b - 1, 1)
        eye = torch.eye(d, device=x.device).unsqueeze(0)
        cov = cov + self.eps * eye
        self.mean = mean
        self.inv_cov = torch.linalg.pinv(cov)
        return self

    def score(self, embeddings: torch.Tensor) -> PadimScores:
        if self.mean is None or self.inv_cov is None:
            raise RuntimeError("PaDiM model is not fitted")
        x = F.normalize(embeddings.float(), dim=-1)
        x = self._select(x)
        mean = self.mean.to(x.device)
        inv_cov = self.inv_cov.to(x.device)
        diff = x - mean.unsqueeze(0)
        patch_scores = torch.einsum("bnd,nde,bne->bn", diff, inv_cov, diff).clamp_min(0).sqrt()
        k = max(1, min(self.top_k, patch_scores.shape[1]))
        image_scores = patch_scores.topk(k, dim=1).values.mean(dim=1)
        return PadimScores(patch_scores=patch_scores, image_scores=image_scores)

    def fit_score(self, train_embeddings: torch.Tensor, test_embeddings: torch.Tensor) -> PadimScores:
        self.fit(train_embeddings)
        return self.score(test_embeddings)
