from __future__ import annotations

from dataclasses import dataclass
import hashlib
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


def split_nominal_train_validation(
    train_records: list[ImageRecord], val_fraction: float = 0.2, seed: int = 42
) -> tuple[list[ImageRecord], list[ImageRecord]]:
    """Create a deterministic nominal-only validation split.

    MVTec AD provides normal train images and a labelled test set, but no
    anomalous validation split.  This holdout is suitable for a transparent
    normal-score quantile threshold; it is not used to tune on test labels.
    """

    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")
    if len(train_records) < 2:
        raise ValueError("at least two nominal training records are required")

    def key(record: ImageRecord) -> str:
        return hashlib.sha256(f"{seed}:{record.image_path}".encode("utf-8")).hexdigest()

    ordered = sorted(train_records, key=key)
    n_val = min(len(ordered) - 1, max(1, int(round(len(ordered) * val_fraction))))
    validation = ordered[:n_val]
    fit = ordered[n_val:]
    assert not ({r.image_path for r in fit} & {r.image_path for r in validation})
    return fit, validation
