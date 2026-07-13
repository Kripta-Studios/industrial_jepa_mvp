import torch
import pytest

from industrial_world_model.visual.feature_extractors import BackboneUnavailableError, build_feature_extractor
from industrial_world_model.visual.dinov3_features import build_dinov3_or_fallback


def test_patch_feature_shape():
    extractor = build_feature_extractor("patch_stats", patch_size=16, image_size=64)
    x = torch.rand(2, 3, 64, 64)
    feats, grid = extractor.extract(x)
    assert feats.shape == (2, 16, 12)
    assert grid == (4, 4)


def test_dinov3_fails_closed_without_gated_authorization(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    with pytest.raises(BackboneUnavailableError, match="gated_weights_unauthorized"):
        build_feature_extractor("dinov3", patch_size=16, image_size=64)


def test_dinov3_fallback_is_explicit_and_executes_named_backend(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    extractor = build_feature_extractor("dinov3", patch_size=16, image_size=64, allow_fallback=True)
    assert extractor.info.requested_backbone == "dinov3"
    assert extractor.info.actual_backbone == "patch_stats"
    assert extractor.info.fallback_used is True
    assert extractor.info.blocker_code == "gated_weights_unauthorized"
    feats, grid = extractor.extract(torch.rand(1, 3, 64, 64))
    assert feats.shape == (1, 16, 12)
    assert grid == (4, 4)


def test_legacy_dinov3_builder_fails_closed_by_default(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    with pytest.raises(BackboneUnavailableError, match="gated_weights_unauthorized"):
        build_dinov3_or_fallback(patch_size=16, image_size=64)


def test_legacy_dinov3_builder_requires_explicit_fallback(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    extractor = build_dinov3_or_fallback(patch_size=16, image_size=64, allow_fallback=True)
    assert extractor.info.requested_backbone == "dinov3"
    assert extractor.info.actual_backbone == "patch_stats"
    assert extractor.info.fallback_used is True
