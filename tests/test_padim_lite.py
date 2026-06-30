import torch

from visual_jepa.models.padim_lite import PadimLite


def test_padim_lite_regularized_scores_shapes():
    train = torch.randn(5, 4, 6)
    test = torch.randn(3, 4, 6)
    padim = PadimLite(n_features=3, eps=1e-2, seed=42, top_k=2).fit(train)
    scores = padim.score(test)

    assert padim.mean is not None
    assert padim.inv_cov is not None
    assert padim.mean.shape == (4, 3)
    assert padim.inv_cov.shape == (4, 3, 3)
    assert scores.patch_scores.shape == (3, 4)
    assert scores.image_scores.shape == (3,)
    assert torch.isfinite(scores.patch_scores).all()

