from __future__ import annotations

from dataclasses import asdict

from .feature_extractors import build_feature_extractor


def build_dinov3_or_fallback(
    patch_size: int = 16,
    image_size: int = 224,
    *,
    allow_fallback: bool = False,
):
    """Build the legacy DINOv3 slot, failing closed unless explicitly allowed.

    The function deliberately avoids network downloads. Reports must inspect
    ``extractor.info`` before claiming which backbone was actually used.
    """

    return build_feature_extractor(
        "dinov3",
        patch_size=patch_size,
        image_size=image_size,
        allow_fallback=allow_fallback,
    )


def backbone_report(extractor) -> dict:
    return asdict(extractor.info)
