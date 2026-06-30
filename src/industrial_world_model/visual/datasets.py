from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass
class ImageRecord:
    image_path: str
    dataset: str
    category: str
    split: str
    label: int
    defect_type: str
    mask_path: str | None = None


def _images(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def collect_mvtec_records(
    root: str | Path,
    dataset: str = "mvtec_ad",
    categories: list[str] | None = None,
    max_samples: int | None = None,
) -> list[ImageRecord]:
    root = Path(root)
    if not root.exists():
        return []
    cats = categories or sorted(p.name for p in root.iterdir() if p.is_dir())
    records: list[ImageRecord] = []
    for cat in cats:
        cat_root = root / cat
        if not cat_root.exists():
            continue
        for split in ["train", "test"]:
            split_root = cat_root / split
            for img in _images(split_root):
                try:
                    defect_type = img.parent.name
                except Exception:
                    defect_type = "unknown"
                label = 0 if defect_type == "good" else 1
                mask_path = None
                if label:
                    candidate = cat_root / "ground_truth" / defect_type / f"{img.stem}_mask.png"
                    if candidate.exists():
                        mask_path = str(candidate)
                records.append(
                    ImageRecord(
                        image_path=str(img),
                        dataset=dataset,
                        category=cat,
                        split=split,
                        label=label,
                        defect_type=defect_type,
                        mask_path=mask_path,
                    )
                )
                if max_samples is not None and len(records) >= max_samples:
                    return records
    return records


def split_records(records: list[ImageRecord]) -> tuple[list[ImageRecord], list[ImageRecord]]:
    train = [r for r in records if r.split == "train" and r.label == 0]
    test = [r for r in records if r.split == "test"]
    return train, test
