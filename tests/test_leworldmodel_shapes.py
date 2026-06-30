import torch
import torch.nn.functional as F

from industrial_world_model.world_model.models import LatentWorldModel


def test_leworldmodel_shapes_and_missing_action():
    model = LatentWorldModel(latent_dim=12, action_dim=4)
    z = torch.randn(5, 12)
    out = model(z, None, horizon=3)
    assert out.shape == (5, 12)
    loss = F.mse_loss(out, torch.randn_like(out))
    assert torch.isfinite(loss)
