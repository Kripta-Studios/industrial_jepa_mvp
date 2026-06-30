import torch

from visual_jepa.models.feature_memory import PatchFeatureMemory
from visual_jepa.models.patchcore_lite import PatchCoreLite


def test_feature_memory_scores_shapes():
    train = torch.randn(4, 6, 8)
    test = torch.randn(3, 6, 8)
    memory = PatchFeatureMemory(coreset_ratio=0.5, top_k=2, seed=42).fit(train)
    scores = memory.score(test)

    assert scores.patch_scores.shape == (3, 6)
    assert scores.image_scores.shape == (3,)
    assert memory.memory is not None
    assert 1 <= len(memory.memory) <= 24


def test_patchcore_lite_fit_score():
    train = torch.randn(4, 5, 6)
    test = torch.randn(2, 5, 6)
    scores = PatchCoreLite(coreset_ratio=1.0, top_k=3).fit_score(train, test)
    assert scores.patch_scores.shape == (2, 5)
    assert scores.image_scores.shape == (2,)

