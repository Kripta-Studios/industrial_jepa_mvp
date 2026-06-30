from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


HARD_SPLIT_ALIASES: dict[str, list[str]] = {
    "held_out_tool_id": ["ToolIndex", "tool_id"],
    "held_out_tool_type": ["MillingToolType", "tool_type"],
    "held_out_hardness_bin": ["HardnessMean", "hardness"],
    "held_out_feed_rate_bin": ["FeedRate", "feed_rate"],
    "held_out_rotation_bin": ["ToolRotation", "rotation", "rpm"],
    "held_out_cutting_condition": ["ADOC", "RDOC"],
    "held_out_holder_length_bin": ["ToolHolderLength", "holder_length"],
}


@dataclass(frozen=True)
class HardSplit:
    name: str
    status: str
    group_column: str | None
    train_mask: np.ndarray
    val_mask: np.ndarray
    test_mask: np.ndarray
    train_groups: list[str]
    val_groups: list[str]
    test_groups: list[str]
    reason: str = ""

    @property
    def passes_no_overlap(self) -> bool:
        train = set(self.train_groups)
        val = set(self.val_groups)
        test = set(self.test_groups)
        return not (train & val or train & test or val & test)


def find_existing_columns(df: pd.DataFrame, aliases: Iterable[str]) -> list[str]:
    lower_to_col = {c.lower(): c for c in df.columns}
    found = []
    for alias in aliases:
        col = lower_to_col.get(alias.lower())
        if col is not None:
            found.append(col)
    return found


def _empty_pending(name: str, n: int, reason: str) -> HardSplit:
    empty = np.zeros(n, dtype=bool)
    return HardSplit(name, "pending", None, empty, empty, empty, [], [], [], reason)


def _numeric_bins(values: pd.Series, bins: int = 3) -> pd.Series:
    clean = pd.to_numeric(values, errors="coerce")
    if clean.nunique(dropna=True) <= 1:
        return pd.Series(["single_bin"] * len(values), index=values.index)
    try:
        return pd.qcut(clean.rank(method="first"), q=min(bins, clean.nunique()), labels=False, duplicates="drop").astype(str)
    except Exception:
        return pd.cut(clean, bins=min(bins, max(2, clean.nunique())), labels=False, duplicates="drop").astype(str)


def make_group_split(
    groups: pd.Series,
    name: str,
    seed: int = 42,
    val_fraction: float = 0.15,
    test_fraction: float = 0.20,
    min_groups: int = 3,
) -> HardSplit:
    groups = groups.astype(str).fillna("missing")
    unique = np.array(sorted(groups.unique().tolist()), dtype=object)
    n = len(groups)
    if len(unique) < min_groups:
        return _empty_pending(name, n, f"need at least {min_groups} groups, found {len(unique)}")
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n_test = max(1, int(round(len(unique) * test_fraction)))
    n_val = max(1, int(round(len(unique) * val_fraction)))
    if n_test + n_val >= len(unique):
        n_val = max(1, len(unique) - n_test - 1)
    test_groups = unique[:n_test].tolist()
    val_groups = unique[n_test : n_test + n_val].tolist()
    train_groups = unique[n_test + n_val :].tolist()
    if not train_groups:
        return _empty_pending(name, n, "split would leave empty train groups")
    train_mask = groups.isin(train_groups).to_numpy()
    val_mask = groups.isin(val_groups).to_numpy()
    test_mask = groups.isin(test_groups).to_numpy()
    return HardSplit(
        name=name,
        status="ok",
        group_column=groups.name,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        train_groups=[str(g) for g in train_groups],
        val_groups=[str(g) for g in val_groups],
        test_groups=[str(g) for g in test_groups],
    )


def build_hard_split(meta: pd.DataFrame, split_name: str, seed: int = 42) -> HardSplit:
    n = len(meta)
    if split_name not in HARD_SPLIT_ALIASES:
        return _empty_pending(split_name, n, f"unknown split name: {split_name}")
    columns = find_existing_columns(meta, HARD_SPLIT_ALIASES[split_name])
    if not columns:
        return _empty_pending(split_name, n, "required column is missing")
    if split_name == "held_out_cutting_condition":
        missing = [c for c in ["ADOC", "RDOC"] if c not in meta.columns]
        if missing:
            return _empty_pending(split_name, n, f"missing cutting-condition columns: {missing}")
        groups = (meta["ADOC"].astype(str) + "_rdoc_" + meta["RDOC"].astype(str)).rename("cutting_condition")
        return make_group_split(groups, split_name, seed=seed)
    column = columns[0]
    if split_name.endswith("_bin"):
        groups = _numeric_bins(meta[column]).rename(f"{column}_bin")
    else:
        groups = meta[column].rename(column)
    return make_group_split(groups, split_name, seed=seed)


def build_all_hard_splits(meta: pd.DataFrame, seed: int = 42) -> list[HardSplit]:
    return [build_hard_split(meta, name, seed=seed) for name in HARD_SPLIT_ALIASES]


def hard_split_report_rows(splits: Iterable[HardSplit]) -> list[dict[str, object]]:
    rows = []
    for split in splits:
        rows.append(
            {
                "split_name": split.name,
                "status": split.status,
                "group_column": split.group_column,
                "train_n": int(split.train_mask.sum()),
                "val_n": int(split.val_mask.sum()),
                "test_n": int(split.test_mask.sum()),
                "train_groups": ",".join(split.train_groups),
                "val_groups": ",".join(split.val_groups),
                "test_groups": ",".join(split.test_groups),
                "passes_no_overlap": split.passes_no_overlap,
                "reason": split.reason,
            }
        )
    return rows
