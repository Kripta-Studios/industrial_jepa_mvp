from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from common.config import get_device_name
from common.paths import ensure_dir
from common.reports import write_json
from visual_jepa.data.industrial import DenseVisualDataBundle, prepare_dense_visual_data
from visual_jepa.models.dense_visual_jepa import build_dense_visual_jepa_from_config
from visual_jepa.models.dino_backbone import build_dino_backbone
from visual_jepa.train.pretrain import build_visual_model_from_config


class ResNetPatchBackbone(nn.Module):
    def __init__(self, name: str = "resnet18", pretrained: bool = True):
        super().__init__()
        import torchvision.models as models

        self.name = name
        self.pretrained = False
        weights = None
        if pretrained:
            try:
                if name == "resnet18":
                    weights = models.ResNet18_Weights.DEFAULT
                elif name == "wide_resnet50":
                    weights = models.Wide_ResNet50_2_Weights.DEFAULT
            except Exception:
                weights = None
        try:
            if name == "resnet18":
                base = models.resnet18(weights=weights)
            elif name == "wide_resnet50":
                base = models.wide_resnet50_2(weights=weights)
            else:
                raise ValueError(name)
            self.pretrained = weights is not None
        except Exception:
            if name == "resnet18":
                base = models.resnet18(weights=None)
            elif name == "wide_resnet50":
                base = models.wide_resnet50_2(weights=None)
            else:
                raise
            self.pretrained = False
        self.features = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool, base.layer1, base.layer2, base.layer3)
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def forward_features(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        fmap = self.features(x)
        b, d, h, w = fmap.shape
        tokens = fmap.flatten(2).transpose(1, 2)
        return F.normalize(tokens.float(), dim=-1), (h, w)


def _dense_checkpoint_path(cfg: dict[str, Any]) -> Path | None:
    out = Path(cfg.get("eval", {}).get("output_dir", "outputs/visual_jepa/dense_pretrain"))
    candidates: list[Path] = []
    explicit = str(cfg.get("outputs", {}).get("checkpoint", "") or "").strip()
    if explicit:
        candidates.append(Path(explicit))
    candidates.extend([out / "checkpoints" / "best.pt", out / "checkpoints" / "latest.pt"])
    for p in candidates:
        if p.is_file():
            return p
    return None


def build_feature_backbone(cfg: dict[str, Any], backbone: str, device: str) -> tuple[Any, dict[str, Any]]:
    if backbone == "dense_visual_jepa":
        model = build_dense_visual_jepa_from_config(cfg).to(device)
        ckpt = _dense_checkpoint_path(cfg)
        pretrained = False
        if ckpt is not None:
            state = torch.load(ckpt, map_location=device, weights_only=False)
            model.load_state_dict(state["model_state"], strict=False)
            pretrained = True
        model.eval()
        return model, {"backbone": backbone, "checkpoint": str(ckpt) if ckpt else "", "pretrained": pretrained}
    if backbone == "current_visual_jepa":
        model = build_visual_model_from_config(cfg).to(device)
        ckpt = Path(cfg.get("outputs", {}).get("checkpoint", ""))
        pretrained = False
        if ckpt.exists():
            state = torch.load(ckpt, map_location=device, weights_only=False)
            model.load_state_dict(state["model_state"], strict=False)
            pretrained = True
        model.eval()
        return model, {"backbone": backbone, "checkpoint": str(ckpt), "pretrained": pretrained}
    if backbone in {"resnet18", "wide_resnet50"}:
        model = ResNetPatchBackbone(backbone, pretrained=True).to(device)
        return model, {"backbone": backbone, "pretrained": model.pretrained}
    if backbone in {"dinov2", "dino", "dinov2_vits14"}:
        model, info = build_dino_backbone("dinov2_vits14", device=device)
        return model, info.__dict__
    raise ValueError(f"Unknown feature backbone: {backbone}")


@torch.no_grad()
def _extract_split(model: Any, dataset, split: str, device: str, batch_size: int, backbone: str) -> dict[str, Any]:
    embeddings, labels, masks, rows = [], [], [], []
    grid_shape = None
    if model is None:
        return {"embeddings": torch.empty(0), "labels": torch.empty(0), "masks": torch.empty(0), "rows": [], "grid_shape": None}
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        x = batch["image"].to(device)
        if backbone == "dense_visual_jepa":
            out = model.encode_dense(x)
            tokens, grid_shape = out["tokens"], out["grid_shape"]
        elif backbone == "current_visual_jepa":
            fmap = model.feature_map(x)
            _, d, h, w = fmap.shape
            tokens = F.normalize(fmap.flatten(2).transpose(1, 2).float(), dim=-1)
            grid_shape = (h, w)
        else:
            tokens, grid_shape = model.forward_features(x)
        embeddings.append(tokens.cpu())
        labels.append(batch["label"].cpu())
        masks.append(batch["mask"].squeeze(1).cpu().to(torch.uint8))
        for i in range(len(x)):
            rows.append(
                {
                    "split": split,
                    "dataset": batch["dataset"][i],
                    "category": batch["category"][i],
                    "label": int(batch["label"][i].item()),
                    "image_path": batch["path"][i],
                    "mask_path": batch["mask_path"][i],
                    "defect_type": batch["defect_type"][i],
                }
            )
    return {
        "embeddings": torch.cat(embeddings, dim=0) if embeddings else torch.empty(0),
        "labels": torch.cat(labels, dim=0) if labels else torch.empty(0, dtype=torch.long),
        "masks": torch.cat(masks, dim=0) if masks else torch.empty(0),
        "rows": rows,
        "grid_shape": grid_shape,
    }


def _save_group(root: Path, backbone: str, split: str, data: dict[str, Any], backbone_info: dict[str, Any]) -> list[Path]:
    paths = []
    if len(data["rows"]) == 0:
        return paths
    df = pd.DataFrame(data["rows"])
    for (dataset, category), idx_df in df.groupby(["dataset", "category"], dropna=False):
        idx = torch.tensor(idx_df.index.to_numpy(), dtype=torch.long)
        out_dir = ensure_dir(root / backbone / str(dataset) / str(category))
        path = out_dir / f"features_{split}.pt"
        torch.save(
            {
                "embeddings": data["embeddings"][idx],
                "labels": data["labels"][idx],
                "masks": data["masks"][idx],
                "metadata": idx_df.to_dict("records"),
                "grid_shape": data["grid_shape"],
                "backbone_info": backbone_info,
            },
            path,
        )
        paths.append(path)
    return paths


def extract_dense_features(cfg: dict[str, Any], backbone: str = "dense_visual_jepa") -> dict[str, Any]:
    device = get_device_name(cfg.get("device", "auto"))
    bundle: DenseVisualDataBundle = prepare_dense_visual_data(cfg)
    batch_size = int(cfg.get("train", {}).get("batch_size", 32))
    root = ensure_dir(cfg.get("features", {}).get("output_dir", "outputs/visual_jepa/features"))
    model, info = build_feature_backbone(cfg, backbone, device)
    paths = []
    manifest_rows = []
    for split, dataset in [("train", bundle.train_dataset), ("val", bundle.val_dataset), ("test", bundle.test_dataset)]:
        data = _extract_split(model, dataset, split, device, batch_size, backbone)
        paths.extend(_save_group(root, backbone, split, data, info))
        manifest_rows.extend(data["rows"])
    manifest = pd.DataFrame(manifest_rows)
    manifest_path = root / backbone / "feature_manifest.csv"
    ensure_dir(manifest_path.parent)
    manifest.to_csv(manifest_path, index=False)
    write_json(root / backbone / "backbone_info.json", info)
    return {"paths": paths, "manifest": manifest_path, "backbone_info": info}
