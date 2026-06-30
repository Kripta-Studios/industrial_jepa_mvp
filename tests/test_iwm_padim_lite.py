import pytest
import torch

from industrial_world_model.visual.padim_lite import PadimLite


@pytest.mark.skipif(PadimLite is None, reason="PaDiM implementation unavailable")
def test_iwm_padim_fit_score():
    train = torch.randn(5, 6, 10)
    test = torch.randn(3, 6, 10)
    scores = PadimLite(n_features=5, eps=1e-2).fit_score(train, test)
    assert scores.patch_scores.shape == (3, 6)
    assert scores.image_scores.shape == (3,)
    assert torch.isfinite(scores.patch_scores).all()
