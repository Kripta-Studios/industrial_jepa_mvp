import torch

from visual_jepa.models.dense_visual_jepa import DenseVisualJEPA


def test_ema_target_encoder_has_no_gradients_and_updates_smoothly():
    model = DenseVisualJEPA(
        image_size=32,
        channels=3,
        patch_size=8,
        embedding_dim=32,
        depth=1,
        num_heads=4,
        predictor="mlp",
        min_block_size=1,
        max_block_size=2,
        target_mode="ema",
        seed=7,
    )

    assert all(not p.requires_grad for p in model.target_encoder.parameters())

    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    out["loss"].backward()
    assert all(p.grad is None for p in model.target_encoder.parameters())

    target_param = next(model.target_encoder.parameters())
    context_param = next(model.context_encoder.parameters())
    before = target_param.detach().clone()
    with torch.no_grad():
        context_param.add_(1.0)
    model.ema_momentum = 0.5
    model.update_target_encoder()
    after = target_param.detach()
    assert not torch.allclose(before, after)
    assert torch.allclose(after, before * 0.5 + context_param.detach() * 0.5)

