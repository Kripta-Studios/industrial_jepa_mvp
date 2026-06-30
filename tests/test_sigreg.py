import torch

from industrial_world_model.lejepa.sigreg import SIGReg, collapse_diagnostics


def test_sigreg_finite_and_has_gradients():
    torch.manual_seed(0)
    z = torch.randn(16, 8, requires_grad=True)
    loss = SIGReg(embedding_dim=8, num_projections=16)(z)
    assert torch.isfinite(loss)
    loss.backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()


def test_sigreg_deterministic_projection():
    a = SIGReg(embedding_dim=8, seed=123).projection
    b = SIGReg(embedding_dim=8, seed=123).projection
    assert torch.allclose(a, b)


def test_collapse_diagnostics_include_rank_and_isotropy():
    z = torch.randn(32, 8)
    logs = collapse_diagnostics(z)
    assert logs["effective_rank"] > 0
    assert 0 <= logs["effective_rank_ratio"] <= 1
    assert logs["isotropy_score"] >= 0
