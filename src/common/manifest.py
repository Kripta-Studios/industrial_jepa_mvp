from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .paths import ensure_dir


SENSOR_DATASETS = [
    "cnc_milling",
    "cwru_bearing",
    "paderborn_bearing",
    "multi_sensor_cnc",
    "nasa_ims_bearing",
    "cmapss",
    "ai4i",
]

VISUAL_DATASETS = [
    "mvtec_ad",
    "visa",
    "kolektor_sdd",
    "mvtec_3d_ad",
    "neu_surface",
    "dagm_2007",
    "severstal",
    "wood_surface_defects",
]


def _count_files(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not path.exists():
        return counts
    for f in path.rglob("*"):
        if f.is_file():
            ext = f.suffix.lower() or "<none>"
            counts[ext] = counts.get(ext, 0) + 1
    return dict(sorted(counts.items()))


def scan_datasets(raw_root: str | Path = "data/raw") -> list[dict[str, Any]]:
    raw_root = Path(raw_root)
    rows: list[dict[str, Any]] = []
    for family, names in [("sensor", SENSOR_DATASETS), ("visual", VISUAL_DATASETS)]:
        for name in names:
            path = raw_root / family / name
            counts = _count_files(path)
            files_total = sum(counts.values())
            status = "missing" if files_total == 0 else "partial"
            if name in {"cnc_milling", "mvtec_ad"} and files_total:
                status = "implemented"
            elif name in {"cwru_bearing", "paderborn_bearing", "visa", "kolektor_sdd"} and files_total:
                status = "partial"
            rows.append(
                {
                    "family": family,
                    "dataset": name,
                    "local_path": str(path).replace("\\", "/"),
                    "status": status,
                    "files_total": files_total,
                    "extensions": counts,
                }
            )
    return rows


def write_dataset_manifest(
    raw_root: str | Path = "data/raw",
    out_yaml: str | Path = "data/manifests/datasets.yaml",
    out_csv: str | Path = "data/manifests/datasets.csv",
) -> tuple[Path, Path]:
    rows = scan_datasets(raw_root)
    out_yaml = Path(out_yaml)
    out_csv = Path(out_csv)
    ensure_dir(out_yaml.parent)
    with out_yaml.open("w", encoding="utf-8") as f:
        yaml.safe_dump({"datasets": rows}, f, sort_keys=False)
    flat = []
    for row in rows:
        r = row.copy()
        r["extensions"] = ";".join(f"{k}:{v}" for k, v in row["extensions"].items())
        flat.append(r)
    pd.DataFrame(flat).to_csv(out_csv, index=False)
    return out_yaml, out_csv

