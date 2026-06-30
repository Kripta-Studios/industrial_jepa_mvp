from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset

from .transforms import load_mask, load_rgb, normalize_image


def build_mvtec_manifest(
    raw_root: str | Path,
    category: str,
    manifest_path: str | Path | None = None,
) -> pd.DataFrame:
    root = Path(raw_root) / category
    if not root.exists():
        raise FileNotFoundError(f"MVTec category not found: {root}")
    rows = []
    for path in sorted((root / "train" / "good").glob("*.png")):
        rows.append({"split": "train", "category": category, "label": 0, "defect_type": "good", "image": str(path), "mask": ""})
    for defect_dir in sorted((root / "test").iterdir()):
        if not defect_dir.is_dir():
            continue
        label = 0 if defect_dir.name == "good" else 1
        for path in sorted(defect_dir.glob("*.png")):
            mask = ""
            if label:
                mask_path = root / "ground_truth" / defect_dir.name / f"{path.stem}_mask.png"
                mask = str(mask_path) if mask_path.exists() else ""
            rows.append(
                {
                    "split": "test",
                    "category": category,
                    "label": label,
                    "defect_type": defect_dir.name,
                    "image": str(path),
                    "mask": mask,
                }
            )
    df = pd.DataFrame(rows)
    if manifest_path is not None:
        manifest_path = Path(manifest_path)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(manifest_path, index=False)
    return df


class MVTecADDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, image_size: int = 128, split: str | None = None, max_images: int | None = None):
        df = manifest.copy()
        if split is not None:
            df = df[df["split"] == split]
        if max_images is not None:
            df = df.head(max_images)
        self.df = df.reset_index(drop=True)
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = normalize_image(load_rgb(row["image"], self.image_size))
        mask_path = row.get("mask", "")
        mask = load_mask(mask_path if isinstance(mask_path, str) and mask_path else None, self.image_size)
        return {
            "image": img,
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
            "mask": mask,
            "path": row["image"],
            "defect_type": row["defect_type"],
        }


@dataclass
class VisualBundle:
    manifest: pd.DataFrame
    train_dataset: MVTecADDataset
    test_dataset: MVTecADDataset


def prepare_mvtec_from_config(cfg: dict[str, Any], force: bool = False) -> VisualBundle:
    data_cfg = cfg["data"]
    manifest_path = Path(data_cfg["manifest_path"])
    if manifest_path.exists() and not force:
        manifest = pd.read_csv(manifest_path)
    else:
        manifest = build_mvtec_manifest(data_cfg["raw_root"], data_cfg["category"], manifest_path)
    image_size = int(data_cfg.get("image_size", 128))
    train = MVTecADDataset(
        manifest,
        image_size=image_size,
        split="train",
        max_images=data_cfg.get("max_train_images"),
    )
    test = MVTecADDataset(
        manifest,
        image_size=image_size,
        split="test",
        max_images=data_cfg.get("max_test_images"),
    )
    return VisualBundle(manifest=manifest, train_dataset=train, test_dataset=test)

