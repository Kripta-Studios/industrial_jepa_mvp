from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
SIGNAL_EXTS = {".csv", ".txt", ".mat", ".npz", ".npy", ".parquet"}


@dataclass
class DatasetManifestEntry:
    dataset: str
    path: str
    exists: bool
    modality: str
    num_images: int = 0
    num_signal_files: int = 0
    categories: str = ""
    train_count: int = 0
    test_count: int = 0
    has_masks: bool = False
    has_actions: bool = False
    has_temporal_sequence: bool = False
    usable_for_visual_anomaly: bool = False
    usable_for_world_model: bool = False
    usable_for_predictive_quality: bool = False
    notes: str = ""


DEFAULT_DATASETS = [
    ("mvtec_ad", "data/raw/visual/mvtec_ad", "visual"),
    ("visa", "data/raw/visual/visa", "visual"),
    ("kolektor_sdd", "data/raw/visual/kolektor_sdd", "visual"),
    ("mvtec_ad_2", "data/raw/visual/mvtec_ad_2", "visual"),
    ("mvtec_loco_ad", "data/raw/visual/mvtec_loco_ad", "visual"),
    ("real_iad", "data/raw/visual/real_iad", "visual"),
    ("mvtec_3d_ad", "data/raw/visual/mvtec_3d_ad", "visual"),
    ("neu_surface_defect", "data/raw/visual/neu_surface", "visual"),
    ("dagm_2007", "data/raw/visual/dagm_2007", "visual"),
    ("severstal", "data/raw/visual/severstal", "visual"),
    ("wood_surface_defects", "data/raw/visual/wood_surface_defects", "visual"),
    ("cnc_milling", "data/raw/sensor/cnc_milling", "sensor"),
    ("cwru_bearing", "data/raw/sensor/cwru_bearing", "sensor"),
    ("paderborn_bearing", "data/raw/sensor/paderborn_bearing", "sensor"),
    ("tennessee_eastman", "data/raw/sensor/tennessee_eastman", "process"),
    ("nasa_cmapss", "data/raw/sensor/cmapss", "process"),
    ("droid", "data/raw/robotics/droid", "robotics"),
    ("bridgedata_v2", "data/raw/robotics/bridgedata_v2", "robotics"),
]


def _count_files(root: Path, exts: set[str]) -> int:
    if not root.exists():
        return 0
    return sum(1 for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def _categories(root: Path) -> list[str]:
    if not root.exists():
        return []
    dirs = [p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")]
    return sorted(dirs)


def _contains_any(root: Path, names: Iterable[str]) -> bool:
    if not root.exists():
        return False
    wanted = {n.lower() for n in names}
    for p in root.rglob("*"):
        if any(n in p.name.lower() for n in wanted):
            return True
    return False


def scan_dataset(name: str, path: str | Path, modality: str) -> DatasetManifestEntry:
    root = Path(path)
    exists = root.exists()
    cats = _categories(root)
    image_count = _count_files(root, IMAGE_EXTS)
    signal_count = _count_files(root, SIGNAL_EXTS)
    train_count = _count_files(root / "train", IMAGE_EXTS) if root.exists() else 0
    test_count = _count_files(root / "test", IMAGE_EXTS) if root.exists() else 0
    if train_count == 0 and cats:
        train_count = sum(_count_files(root / c / "train", IMAGE_EXTS) for c in cats)
        test_count = sum(_count_files(root / c / "test", IMAGE_EXTS) for c in cats)

    has_masks = _contains_any(root, ["ground_truth", "mask", "masks"])
    has_actions = _contains_any(root, ["action", "setpoint", "recipe", "feed", "speed", "rpm"])
    has_temporal = modality in {"sensor", "process", "robotics"} and signal_count > 0

    visual_ok = modality == "visual" and image_count > 0
    world_ok = has_temporal or has_actions
    pq_ok = visual_ok or has_temporal
    notes = "available" if exists else "missing; not downloaded"
    if modality == "visual" and exists and train_count == 0:
        notes += "; split layout not inferred"
    if modality in {"sensor", "process"} and exists and not has_actions:
        notes += "; actions/setpoints not inferred"

    return DatasetManifestEntry(
        dataset=name,
        path=str(root),
        exists=exists,
        modality=modality,
        num_images=image_count,
        num_signal_files=signal_count,
        categories=", ".join(cats[:30]) + (" ..." if len(cats) > 30 else ""),
        train_count=train_count,
        test_count=test_count,
        has_masks=has_masks,
        has_actions=has_actions,
        has_temporal_sequence=has_temporal,
        usable_for_visual_anomaly=visual_ok,
        usable_for_world_model=world_ok,
        usable_for_predictive_quality=pq_ok,
        notes=notes,
    )


def build_manifest(datasets: list[tuple[str, str, str]] | None = None) -> list[DatasetManifestEntry]:
    return [scan_dataset(*spec) for spec in (datasets or DEFAULT_DATASETS)]


def save_manifest(entries: list[DatasetManifestEntry], out_json: str | Path, out_md: str | Path) -> None:
    out_json = Path(out_json)
    out_md = Path(out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(e) for e in entries]
    out_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    headers = [
        "dataset",
        "exists",
        "modality",
        "categories",
        "has_masks",
        "has_actions",
        "has_temporal_sequence",
        "usable_for_visual_anomaly",
        "usable_for_world_model",
        "notes",
    ]
    lines = ["# Industrial World Model Dataset Manifest", "", "|" + "|".join(headers) + "|", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("|" + "|".join(str(row.get(h, "")).replace("|", "/") for h in headers) + "|")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
