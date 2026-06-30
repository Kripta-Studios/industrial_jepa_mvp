from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FeatureExtractorInfo:
    requested_backbone: str
    actual_backbone: str
    pretrained: bool
    dinov3_available: bool
    fallback_used: bool
    notes: str


class PatchStatsFeatureExtractor:
    """Dependency-light dense patch feature extractor.

    It is not meant to compete with DINO; it is a deterministic fallback that
    keeps the product pipeline executable when pretrained dense features are
    unavailable.
    """

    def __init__(self, patch_size: int = 16, image_size: int = 224):
        self.patch_size = int(patch_size)
        self.image_size = int(image_size)
        self.info = FeatureExtractorInfo(
            requested_backbone="patch_stats",
            actual_backbone="patch_stats",
            pretrained=False,
            dinov3_available=False,
            fallback_used=False,
            notes="RGB patch mean/std/min/max fallback features",
        )

    def extract(self, images: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        if images.ndim != 4:
            raise ValueError("Expected images [B,C,H,W]")
        x = F.interpolate(images.float(), size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        p = self.patch_size
        patches = x.unfold(2, p, p).unfold(3, p, p)
        b, c, gh, gw, _, _ = patches.shape
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(b, gh * gw, c, p * p)
        mean = patches.mean(dim=-1)
        std = patches.std(dim=-1)
        mn = patches.amin(dim=-1)
        mx = patches.amax(dim=-1)
        feats = torch.cat([mean, std, mn, mx], dim=-1)
        return F.normalize(feats, dim=-1), (gh, gw)


class TorchvisionResNetPatchExtractor(PatchStatsFeatureExtractor):
    """Dense ResNet feature extractor with safe fallback semantics."""

    def __init__(self, requested_backbone: str = "resnet18", patch_size: int = 16, image_size: int = 224):
        super().__init__(patch_size=patch_size, image_size=image_size)
        self.requested_backbone = requested_backbone
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: nn.Module | None = None
        self.info = FeatureExtractorInfo(
            requested_backbone=requested_backbone,
            actual_backbone="patch_stats_resnet_safe_fallback",
            pretrained=False,
            dinov3_available=False,
            fallback_used=True,
            notes="ResNet/DINO production slot; patch_stats fallback used without downloading weights",
        )
        self._load()

    def _load(self) -> None:
        try:
            import torchvision.models as models

            if self.requested_backbone == "wide_resnet50":
                weights = models.Wide_ResNet50_2_Weights.DEFAULT
                backbone = models.wide_resnet50_2(weights=weights)
                channels = 1024
            else:
                weights = models.ResNet18_Weights.DEFAULT
                backbone = models.resnet18(weights=weights)
                channels = 256
            self.model = nn.Sequential(
                backbone.conv1,
                backbone.bn1,
                backbone.relu,
                backbone.maxpool,
                backbone.layer1,
                backbone.layer2,
                backbone.layer3,
            ).eval().to(self.device)
            for param in self.model.parameters():
                param.requires_grad_(False)
            self.info = FeatureExtractorInfo(
                requested_backbone=self.requested_backbone,
                actual_backbone=f"torchvision_{self.requested_backbone}_layer3",
                pretrained=True,
                dinov3_available=False,
                fallback_used=False,
                notes=f"Pretrained torchvision {self.requested_backbone} layer3 dense features ({channels} channels)",
            )
        except Exception as exc:
            self.model = None
            self.info = FeatureExtractorInfo(
                requested_backbone=self.requested_backbone,
                actual_backbone="patch_stats_resnet_safe_fallback",
                pretrained=False,
                dinov3_available=False,
                fallback_used=True,
                notes=f"Pretrained ResNet unavailable, patch_stats fallback used: {type(exc).__name__}: {exc}",
            )

    @staticmethod
    def _normalize(x: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        return (x - mean) / std

    @torch.no_grad()
    def extract(self, images: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        if self.model is None:
            return super().extract(images)
        if images.ndim != 4:
            raise ValueError("Expected images [B,C,H,W]")
        x = F.interpolate(images.float(), size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        feats = self.model(self._normalize(x.to(self.device))).detach().float().cpu()
        b, c, gh, gw = feats.shape
        tokens = feats.permute(0, 2, 3, 1).reshape(b, gh * gw, c)
        return F.normalize(tokens, dim=-1), (gh, gw)


class DinoV2FeatureExtractor:
    def __init__(self, model_name: str = "dinov2_vits14", image_size: int = 224, batch_size: int = 8):
        self.model_name = model_name
        self.image_size = int(image_size)
        self.batch_size = int(batch_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.info = FeatureExtractorInfo(
            requested_backbone="dinov2",
            actual_backbone="patch_stats_dinov2_safe_fallback",
            pretrained=False,
            dinov3_available=False,
            fallback_used=True,
            notes="DINOv2 not loaded yet",
        )
        self._fallback = PatchStatsFeatureExtractor(patch_size=16, image_size=image_size)
        self._load()

    def _load(self) -> None:
        try:
            import pathlib

            hub_dir = pathlib.Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"
            if not hub_dir.exists():
                raise FileNotFoundError(f"DINOv2 torch hub cache not found: {hub_dir}")
            model = torch.hub.load(str(hub_dir), self.model_name, source="local", pretrained=True)
            model.eval().to(self.device)
            for p in model.parameters():
                p.requires_grad_(False)
            self.model = model
            self.info = FeatureExtractorInfo(
                requested_backbone="dinov2",
                actual_backbone=self.model_name,
                pretrained=True,
                dinov3_available=False,
                fallback_used=False,
                notes="DINOv2 loaded from local torch.hub cache",
            )
        except Exception as exc:
            self.info = FeatureExtractorInfo(
                requested_backbone="dinov2",
                actual_backbone="patch_stats_dinov2_safe_fallback",
                pretrained=False,
                dinov3_available=False,
                fallback_used=True,
                notes=f"DINOv2 unavailable, patch_stats fallback used: {type(exc).__name__}: {exc}",
            )

    @staticmethod
    def _normalize(x: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        return (x - mean) / std

    @torch.no_grad()
    def extract(self, images: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        if self.model is None:
            return self._fallback.extract(images)
        x = F.interpolate(images.float(), size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        feats = []
        grid_shape = (self.image_size // 14, self.image_size // 14)
        for start in range(0, x.shape[0], self.batch_size):
            batch = self._normalize(x[start : start + self.batch_size].to(self.device))
            out = self.model.forward_features(batch)
            tokens = out.get("x_norm_patchtokens") if isinstance(out, dict) else out
            if tokens is None:
                raise RuntimeError("DINOv2 did not return patch tokens")
            feats.append(F.normalize(tokens.detach().float().cpu(), dim=-1))
        return torch.cat(feats, dim=0), grid_shape


def build_feature_extractor(backbone: str = "patch_stats", patch_size: int = 16, image_size: int = 224):
    name = backbone.lower()
    if name in {"patch_stats", "pixel_patch", "pixel"}:
        return PatchStatsFeatureExtractor(patch_size=patch_size, image_size=image_size)
    if name in {"dinov2", "dino"}:
        return DinoV2FeatureExtractor(image_size=image_size)
    if name in {"resnet18", "wide_resnet50", "dinov3"}:
        extractor = TorchvisionResNetPatchExtractor(requested_backbone=backbone, patch_size=patch_size, image_size=image_size)
        if name == "dinov3":
            extractor.info = FeatureExtractorInfo(
                requested_backbone="dinov3",
                actual_backbone="patch_stats_dinov3_safe_fallback",
                pretrained=False,
                dinov3_available=False,
                fallback_used=True,
                notes="DINOv3 weights were not loaded automatically; fallback keeps benchmark executable",
            )
        return extractor
    raise ValueError(f"Unknown backbone: {backbone}")
