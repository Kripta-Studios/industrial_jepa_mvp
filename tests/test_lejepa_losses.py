import torch

from industrial_world_model.lejepa.losses import lejepa_loss
from industrial_world_model.lejepa.sigreg import SIGReg, collapse_diagnostics


def test_lejepa_loss_finite():
    pred = torch.randn(8, 16, requires_grad=True)
    target = torch.randn(8, 16)
    sigreg = SIGReg(embedding_dim=16)
    loss, logs = lejepa_loss(pred, target, torch.cat([pred.detach(), target], dim=0), sigreg)
    assert torch.isfinite(loss)
    assert logs["loss"] > 0


def test_collapse_diagnostics_detect_constant_embeddings():
    z = torch.ones(16, 8)
    logs = collapse_diagnostics(z)
    assert logs["collapse_flag"] is True
