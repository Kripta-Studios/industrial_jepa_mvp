from __future__ import annotations

from pathlib import Path

from scipy.io import loadmat


def inspect_cwru_mat(path: str | Path) -> dict:
    md = loadmat(path, squeeze_me=True, struct_as_record=False)
    keys = [k for k in md.keys() if not k.startswith("__")]
    return {"path": str(path), "keys": keys}

