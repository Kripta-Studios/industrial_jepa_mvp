import torch

from industrial_world_model.visual.patchcore import PatchCoreLite


def test_patchcore_scores_shape_and_finite():
    train = torch.randn(4, 9, 8)
    test = torch.randn(3, 9, 8)
    scores = PatchCoreLite(coreset_ratio=1.0, top_k=2).fit_score(train, test)
    assert scores.patch_scores.shape == (3, 9)
    assert scores.image_scores.shape == (3,)
    assert torch.isfinite(scores.image_scores).all()
