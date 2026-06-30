import torch

from sensor_jepa.models.token_world_model import TokenWorldModel, token_surprise


def test_token_world_model_shapes_and_surprise():
    model = TokenWorldModel(embedding_dim=16, action_dim=3, hidden_dim=32)
    z = torch.randn(5, 7, 16)
    a = torch.randn(5, 3)
    pred = model(z, a, horizon=3)
    assert pred.shape == z.shape
    scores = token_surprise(pred, z)
    assert scores.shape == (5, 7)
    assert torch.isfinite(scores).all()


def test_token_world_model_no_action():
    model = TokenWorldModel(embedding_dim=8, action_dim=0, hidden_dim=16)
    z = torch.randn(2, 4, 8)
    pred = model(z, None, horizon=1)
    assert pred.shape == z.shape
