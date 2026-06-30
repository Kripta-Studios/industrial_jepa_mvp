from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_kolektor_manifest(raw_root: str | Path) -> pd.DataFrame:
    root = Path(raw_root) / "KolektorSDD-boxes"
    rows = []
    for img in sorted(root.rglob("*.jpg")):
        mask = img.with_name(f"{img.stem}_label.bmp")
        rows.append({"image": str(img), "mask": str(mask), "label": int(mask.exists()), "group": img.parent.name})
    return pd.DataFrame(rows)

