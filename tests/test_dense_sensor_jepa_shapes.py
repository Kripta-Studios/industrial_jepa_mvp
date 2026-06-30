import torch

from sensor_jepa.models.dense_sensor_jepa import DenseSensorJEPA


def test_dense_sensor_jepa_forward_shapes_and_no_nan():
    model = DenseSensorJEPA(
        input_channels=4,
        embedding_dim=32,
        depth=1,
        num_heads=4,
        temporal_patch_size=2,
        temporal_patch_stride=2,
        predictor_depth=1,
        temporal_mask_ratio=0.5,
        target_mode="ema",
        sigreg_weight=0.01,
    )
    x = torch.randn(3, 8, 4)
    out = model(x)
    assert out["pred_target"].shape[0] == 3
    assert out["pred_target"].shape[-1] == 32
    assert out["target_embeddings"].shape == out["pred_target"].shape
    assert torch.isfinite(out["loss"])
    assert torch.isfinite(out["sigreg_loss"])
    assert "effective_rank_ratio" in out
    out["loss"].backward()
    assert all(param.grad is None for param in model.target_encoder.parameters())


def test_dense_sensor_jepa_ema_update_changes_target_smoothly():
    model = DenseSensorJEPA(
        input_channels=2,
        embedding_dim=16,
        depth=1,
        num_heads=4,
        temporal_patch_size=2,
        temporal_patch_stride=1,
        predictor_depth=1,
        target_mode="ema",
        ema_momentum=0.5,
    )
    before = next(model.target_encoder.parameters()).detach().clone()
    with torch.no_grad():
        next(model.context_encoder.parameters()).add_(1.0)
    model.update_target_encoder()
    after = next(model.target_encoder.parameters()).detach()
    assert not torch.equal(before, after)
