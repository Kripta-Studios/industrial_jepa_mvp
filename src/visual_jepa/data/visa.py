from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_visa_manifest(raw_root: str | Path, split_csv: str = "1cls.csv") -> pd.DataFrame:
    root = Path(raw_root)
    path = root / "split_csv" / split_csv
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df["image"] = df["image"].map(lambda p: str(root / p))
    df["mask"] = df["mask"].fillna("").map(lambda p: str(root / p) if p else "")
    df["label_binary"] = (df["label"] != "normal").astype(int)
    return df

