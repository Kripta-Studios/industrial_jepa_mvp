import torch

from visual_jepa.models.dense_losses import (
    collapse_metrics,
    covariance_regularization,
    masked_latent_loss,
    variance_regularization,
    visible_latent_loss,
)


def test_dense_losses_are_finite_and_positive():
    pred = torch.randn(2, 3, 8)
    target = torch.randn(2, 3, 8)
    valid = torch.tensor([[True, True, False], [True, False, False]])

    masked = masked_latent_loss(pred, target, valid)
    visible = visible_latent_loss(pred, target, valid)
    cov = covariance_regularization(pred)

    assert torch.isfinite(masked)
    assert torch.isfinite(visible)
    assert torch.isfinite(cov)
    assert masked.item() >= 0.0
    assert visible.item() >= 0.0
    assert cov.item() >= 0.0


def test_variance_regularization_detects_collapse():
    collapsed = torch.ones(4, 5, 8)
    diverse = torch.randn(4, 5, 8)

    assert variance_regularization(collapsed).item() > variance_regularization(diverse).item()
    metrics = collapse_metrics(collapsed)
    assert metrics["collapse_score"].item() == 1.0

