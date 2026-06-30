from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset

from .kolektor_sdd import build_kolektor_manifest
from .mvtec_ad import build_mvtec_manifest
from .transforms import load_mask, load_rgb, normalize_image
from .visa import build_visa_manifest


MVTEC_DEFAULT_CATEGORIES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]


class IndustrialVisualDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, image_size: int = 224, max_images: int | None = None):
        df = manifest.copy().reset_index(drop=True)
        if max_images is not None:
            df = df.head(int(max_images))
        self.df = df.reset_index(drop=True)
        self.image_size = int(image_size)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        image = normalize_image(load_rgb(row["image"], self.image_size))
        mask_path = row.get("mask", "")
        mask = load_mask(mask_path if isinstance(mask_path, str) and mask_path else None, self.image_size)
        return {
            "image": image,
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
            "mask": mask,
            "path": str(row["image"]),
            "mask_path": str(mask_path) if isinstance(mask_path, str) else "",
            "dataset": str(row["dataset"]),
            "category": str(row["category"]),
            "split": str(row["split"]),
            "defect_type": str(row.get("defect_type", "")),
        }


@dataclass
class DenseVisualDataBundle:
    manifest: pd.DataFrame
    train_dataset: IndustrialVisualDataset
    val_dataset: IndustrialVisualDataset
    test_dataset: IndustrialVisualDataset


def _as_list(value: Any, available: list[str] | None = None) -> list[str]:
    if value is None or value == "all":
        return available or []
    if isinstance(value, str):
        return [value]
    return list(value)


def _mvtec_rows(entry: dict[str, Any]) -> pd.DataFrame:
    root = Path(entry["root"])
    available = [p.name for p in sorted(root.iterdir()) if p.is_dir()] if root.exists() else []
    cats = _as_list(entry.get("categories", "all"), available or MVTEC_DEFAULT_CATEGORIES)
    frames = []
    for cat in cats:
        try:
            df = build_mvtec_manifest(root, cat)
        except FileNotFoundError:
            continue
        df["dataset"] = "mvtec_ad"
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _visa_rows(entry: dict[str, Any]) -> pd.DataFrame:
    try:
        df = build_visa_manifest(entry["root"])
    except Exception:
        return pd.DataFrame()
    out = pd.DataFrame(
        {
            "dataset": "visa",
            "category": df.get("object", df.get("category", "visa")),
            "split": df.get("split", "train"),
            "label": df.get("label_binary", 0),
            "defect_type": df.get("label", ""),
            "image": df["image"],
            "mask": df.get("mask", ""),
        }
    )
    return out


def _kolektor_rows(entry: dict[str, Any]) -> pd.DataFrame:
    try:
        df = build_kolektor_manifest(entry["root"])
    except Exception:
        return pd.DataFrame()
    split = ["train" if i % 5 else "test" for i in range(len(df))]
    return pd.DataFrame(
        {
            "dataset": "kolektor_sdd",
            "category": df.get("group", "kolektor"),
            "split": split,
            "label": df["label"],
            "defect_type": df["label"].map(lambda v: "defect" if int(v) else "good"),
            "image": df["image"],
            "mask": df.get("mask", ""),
        }
    )


def build_dense_visual_manifest(cfg: dict[str, Any]) -> pd.DataFrame:
    dataset_cfg = cfg["dataset"]
    frames = []
    for entry in dataset_cfg.get("datasets", []):
        if not entry.get("enabled", True):
            continue
        name = entry["name"]
        if name == "mvtec_ad":
            df = _mvtec_rows(entry)
        elif name == "visa":
            df = _visa_rows(entry)
        elif name == "kolektor_sdd":
            df = _kolektor_rows(entry)
        else:
            df = pd.DataFrame()
        if len(df) and entry.get("normal_only", False):
            train_mask = df["split"].eq(entry.get("train_split", "train"))
            df = pd.concat([df[~train_mask], df[train_mask & df["label"].eq(0)]], ignore_index=True)
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No enabled visual datasets could be loaded")
    manifest = pd.concat(frames, ignore_index=True)
    manifest = manifest[["dataset", "category", "split", "label", "defect_type", "image", "mask"]].fillna("")
    return manifest.reset_index(drop=True)


def prepare_dense_visual_data(cfg: dict[str, Any]) -> DenseVisualDataBundle:
    dataset_cfg = cfg["dataset"]
    image_size = int(dataset_cfg.get("image_size", 224))
    manifest = build_dense_visual_manifest(cfg)
    val_ratio = float(dataset_cfg.get("val_ratio", 0.15))
    max_train = dataset_cfg.get("max_train_images")
    max_val = dataset_cfg.get("max_val_images")
    max_test = dataset_cfg.get("max_test_images")
    train_all = manifest[manifest["split"].eq("train") & manifest["label"].eq(0)].copy()
    val_parts = []
    train_parts = []
    for _, group in train_all.groupby(["dataset", "category"], dropna=False):
        group = group.sort_values("image").reset_index(drop=True)
        n_val = max(1, int(round(len(group) * val_ratio))) if len(group) > 1 else 0
        val_parts.append(group.head(n_val))
        train_parts.append(group.iloc[n_val:])
    train_df = pd.concat(train_parts, ignore_index=True) if train_parts else train_all
    val_df = pd.concat(val_parts, ignore_index=True) if val_parts else train_all.head(0)
    test_df = manifest[manifest["split"].eq("test")].copy()
    if test_df.empty:
        test_df = manifest[~manifest.index.isin(train_all.index)].copy()
    return DenseVisualDataBundle(
        manifest=manifest,
        train_dataset=IndustrialVisualDataset(train_df, image_size=image_size, max_images=max_train),
        val_dataset=IndustrialVisualDataset(val_df, image_size=image_size, max_images=max_val),
        test_dataset=IndustrialVisualDataset(test_df, image_size=image_size, max_images=max_test),
    )
