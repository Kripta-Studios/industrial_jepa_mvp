import torch

from industrial_world_model.world_model.surprise import (
    latent_surprise,
    normalize_surprise_by_group,
    residual_surprise,
    temporal_surprise_profile,
)


def test_surprise_score_shape():
    z = torch.randn(7, 8)
    scores = latent_surprise(z + 0.1, z)
    assert scores.shape == (7,)
    assert torch.isfinite(scores).all()


def test_group_normalized_surprise():
    scores = torch.tensor([1.0, 2.0, 10.0, 12.0])
    norm = normalize_surprise_by_group(scores, ["a", "a", "b", "b"])
    assert norm.shape == scores.shape
    assert torch.isfinite(norm).all()


def test_temporal_surprise_profile_and_residual():
    scores = torch.tensor([0.1, 0.2, 0.8, 1.0])
    profile = temporal_surprise_profile(scores, top_k=2)
    assert set(profile) == {"mean", "max", "topk_mean", "ewma_last"}
    assert torch.isclose(profile["topk_mean"], torch.tensor(0.9))
    residual = residual_surprise(scores, torch.tensor([0.0, 0.2, 0.4, 0.6]))
    assert residual.shape == scores.shape
    assert torch.isfinite(residual).all()
