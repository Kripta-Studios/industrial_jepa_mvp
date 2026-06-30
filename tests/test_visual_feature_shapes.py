import torch

from industrial_world_model.visual.feature_extractors import build_feature_extractor


def test_patch_feature_shape():
    extractor = build_feature_extractor("patch_stats", patch_size=16, image_size=64)
    x = torch.rand(2, 3, 64, 64)
    feats, grid = extractor.extract(x)
    assert feats.shape == (2, 16, 12)
    assert grid == (4, 4)


def test_dinov3_fallback_reports_itself():
    extractor = build_feature_extractor("dinov3", patch_size=16, image_size=64)
    assert extractor.info.requested_backbone == "dinov3"
    assert extractor.info.fallback_used is True
