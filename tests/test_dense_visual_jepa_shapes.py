import torch

from visual_jepa.models.dense_visual_jepa import DenseVisualJEPA
from visual_jepa.train.extract_dense_features import _dense_checkpoint_path


def _small_dense_model() -> DenseVisualJEPA:
    return DenseVisualJEPA(
        image_size=32,
        channels=3,
        encoder="vit_tiny",
        patch_size=8,
        embedding_dim=32,
        depth=1,
        num_heads=4,
        predictor="mlp",
        predictor_depth=1,
        predictor_heads=4,
        min_block_size=1,
        max_block_size=2,
        num_target_blocks=2,
        visible_loss_weight=0.1,
        deep_supervision_weight=0.0,
        variance_weight=0.01,
        target_mode="ema",
        seed=42,
    )


def test_dense_visual_jepa_forward_shapes():
    model = _small_dense_model()
    x = torch.randn(2, 3, 32, 32)
    out = model(x)

    assert out["pred_target"].shape == out["target_tokens"].shape
    assert out["pred_target"].shape[0] == 2
    assert out["pred_target"].shape[-1] == 32
    assert out["target_valid"].shape[:2] == out["pred_target"].shape[:2]
    assert out["grid_shape"] == (4, 4)
    assert torch.isfinite(out["loss"])


def test_dense_visual_jepa_encode_and_latent_error_map():
    model = _small_dense_model()
    x = torch.randn(2, 3, 32, 32)
    encoded = model.encode_dense(x)
    assert encoded["tokens"].shape == (2, 16, 32)
    assert encoded["grid_shape"] == (4, 4)

    heat, grid = model.latent_error_map(x)
    assert heat.shape == (2, 1, 4, 4)
    assert grid == (4, 4)


def test_dense_checkpoint_path_ignores_empty_checkpoint(tmp_path):
    cfg = {"outputs": {"checkpoint": ""}, "eval": {"output_dir": str(tmp_path / "missing")}}
    assert _dense_checkpoint_path(cfg) is None
