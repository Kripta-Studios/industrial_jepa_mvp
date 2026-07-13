from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
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

IDENTITY_GROUP_ALIASES: dict[str, list[str]] = {
    "part_id": ["part_id", "part", "workpiece_id", "workpiece", "piece_id", "piece"],
    "tool_id": ["ToolIndex", "tool_id", "tool"],
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
    physical_column: str | None = None
    train_physical_values: list[str] = field(default_factory=list)
    val_physical_values: list[str] = field(default_factory=list)
    test_physical_values: list[str] = field(default_factory=list)

    @property
    def passes_no_overlap(self) -> bool:
        train = set(self.train_groups)
        val = set(self.val_groups)
        test = set(self.test_groups)
        return not (train & val or train & test or val & test)

    @property
    def passes_physical_no_overlap(self) -> bool:
        """Whether the raw physical values, not merely derived bins, are disjoint."""

        train = set(self.train_physical_values)
        val = set(self.val_physical_values)
        test = set(self.test_physical_values)
        return not (train & val or train & test or val & test)


def find_existing_columns(df: pd.DataFrame, aliases: Iterable[str]) -> list[str]:
    lower_to_col = {c.lower(): c for c in df.columns}
    found = []
    for alias in aliases:
        col = lower_to_col.get(alias.lower())
        if col is not None:
            found.append(col)
    return found


def identity_group_series(meta: pd.DataFrame) -> dict[str, pd.Series]:
    """Return physical identity keys that must never cross a partition.

    The CNC source has no explicit part column, but its documented filename
    convention starts with a physical-piece token such as ``P003``.  We retain
    the full filename as a sample identity audit and derive that piece token.
    Temporal indices (for example ``source_cycle``) are not physical IDs and
    are deliberately excluded.
    """

    groups: dict[str, pd.Series] = {}
    for semantic_name, aliases in IDENTITY_GROUP_ALIASES.items():
        for column in find_existing_columns(meta, aliases):
            groups[f"{semantic_name}:{column}"] = meta[column].astype("string").fillna("missing").astype(str)
    filename_columns = find_existing_columns(meta, ["FileName", "file_name", "filename"])
    for column in filename_columns:
        filenames = meta[column].astype("string").fillna("missing").astype(str)
        groups[f"sample_id:{column}"] = filenames
        pieces = filenames.str.extract(r"^([Pp][A-Za-z0-9-]+)(?:_|$)", expand=False)
        if pieces.notna().any():
            groups[f"part_id:{column}_prefix"] = pieces.fillna(filenames)
    return groups


def _connected_component_groups(meta: pd.DataFrame, physical_column: str) -> pd.Series:
    """Join rows linked by an equal target value or any physical identity key."""

    n = len(meta)
    parent = np.arange(n, dtype=int)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    keys = {f"physical:{physical_column}": meta[physical_column], **identity_group_series(meta)}
    for values in keys.values():
        canonical = values.astype("string").fillna("missing").astype(str)
        for indices in canonical.groupby(canonical, sort=False).groups.values():
            rows = [int(index) for index in indices]
            for row in rows[1:]:
                union(rows[0], row)
    roots = [find(index) for index in range(n)]
    labels = {root: f"component_{rank:03d}" for rank, root in enumerate(sorted(set(roots)))}
    return pd.Series([labels[root] for root in roots], index=meta.index, name=f"{physical_column}_identity_component")


def audit_identity_group_overlap(meta: pd.DataFrame, split: "HardSplit") -> dict[str, dict[str, object]]:
    """Audit and fingerprint all available physical/sample identity aliases."""

    audits: dict[str, dict[str, object]] = {}
    for name, values in identity_group_series(meta).items():
        canonical = values.astype("string").fillna("missing").astype(str)
        partitions = {
            "train": sorted(canonical[split.train_mask].unique().tolist()),
            "validation": sorted(canonical[split.val_mask].unique().tolist()),
            "test": sorted(canonical[split.test_mask].unique().tolist()),
        }
        train, validation, test = map(set, partitions.values())
        intersections = {
            "train_validation": sorted(train & validation),
            "train_test": sorted(train & test),
            "validation_test": sorted(validation & test),
        }
        canonical_json = json.dumps(partitions, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        audits[name] = {
            **partitions,
            "intersections": intersections,
            "passes_no_overlap": not any(intersections.values()),
            "sha256": hashlib.sha256(canonical_json.encode("utf-8")).hexdigest(),
        }
    return audits


def _empty_pending(name: str, n: int, reason: str) -> HardSplit:
    empty = np.zeros(n, dtype=bool)
    return HardSplit(name, "pending", None, empty, empty, empty, [], [], [], reason)


def _numeric_bins(values: pd.Series, bins: int = 3) -> pd.Series:
    """Assign equal-frequency bins without ever separating an equal raw value.

    ``rank(method="first")`` ranks rows and can put two measurements of the
    same physical value in different bins.  We bin the sorted *unique* values
    and map that lookup back to every row instead.
    """

    clean = pd.to_numeric(values, errors="coerce")
    unique_values = np.sort(clean.dropna().unique())
    if len(unique_values) <= 1:
        return pd.Series(["single_bin"] * len(values), index=values.index)
    try:
        unique_bins = pd.qcut(
            pd.Series(unique_values),
            q=min(bins, len(unique_values)),
            labels=False,
            duplicates="drop",
        )
    except Exception:
        unique_bins = pd.cut(
            pd.Series(unique_values),
            bins=min(bins, max(2, len(unique_values))),
            labels=False,
            duplicates="drop",
        )
    lookup = {float(value): f"bin_{int(bin_id)}" for value, bin_id in zip(unique_values, unique_bins)}
    out = clean.map(lookup).astype("string").fillna("missing")
    return out.astype(str)


def _canonical_values(values: pd.Series, mask: np.ndarray) -> list[str]:
    numeric = pd.to_numeric(values[mask], errors="coerce")
    if numeric.notna().all():
        return [format(float(v), ".12g") for v in sorted(numeric.unique())]
    return sorted(values[mask].astype("string").fillna("missing").astype(str).unique().tolist())


def _attach_physical_values(split: HardSplit, values: pd.Series, column: str) -> HardSplit:
    return replace(
        split,
        physical_column=column,
        train_physical_values=_canonical_values(values, split.train_mask),
        val_physical_values=_canonical_values(values, split.val_mask),
        test_physical_values=_canonical_values(values, split.test_mask),
    )


def hard_split_fingerprints(split: HardSplit) -> dict[str, str]:
    """Content hashes for exact membership, derived groups and raw values."""

    def digest(payload: object) -> str:
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    masks = {
        "train": np.flatnonzero(split.train_mask).astype(int).tolist(),
        "validation": np.flatnonzero(split.val_mask).astype(int).tolist(),
        "test": np.flatnonzero(split.test_mask).astype(int).tolist(),
    }
    groups = {
        "train": sorted(split.train_groups),
        "validation": sorted(split.val_groups),
        "test": sorted(split.test_groups),
    }
    values = {
        "physical_column": split.physical_column,
        "train": sorted(split.train_physical_values),
        "validation": sorted(split.val_physical_values),
        "test": sorted(split.test_physical_values),
    }
    return {
        "membership_sha256": digest(masks),
        "groups_sha256": digest(groups),
        "physical_values_sha256": digest(values),
        "split_sha256": digest({"name": split.name, "masks": masks, "groups": groups, "values": values}),
    }


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
        split = make_group_split(groups, split_name, seed=seed)
        return _attach_physical_values(split, groups, "ADOC+RDOC") if split.status == "ok" else split
    column = columns[0]
    if split_name == "held_out_hardness_bin":
        # Raw hardness alone is insufficient: the same physical tool could
        # otherwise appear in two partitions at different hardness values.
        # Connected components make every raw value, part and tool atomic.
        groups = _connected_component_groups(meta, column)
    elif split_name.endswith("_bin"):
        groups = _numeric_bins(meta[column]).rename(f"{column}_bin")
    else:
        groups = meta[column].rename(column)
    split = make_group_split(groups, split_name, seed=seed)
    return _attach_physical_values(split, meta[column], column) if split.status == "ok" else split


def build_all_hard_splits(meta: pd.DataFrame, seed: int = 42) -> list[HardSplit]:
    return [build_hard_split(meta, name, seed=seed) for name in HARD_SPLIT_ALIASES]


def hard_split_report_rows(splits: Iterable[HardSplit]) -> list[dict[str, object]]:
    rows = []
    for split in splits:
        fingerprints = hard_split_fingerprints(split)
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
                "physical_column": split.physical_column,
                "train_physical_values": ",".join(split.train_physical_values),
                "val_physical_values": ",".join(split.val_physical_values),
                "test_physical_values": ",".join(split.test_physical_values),
                "passes_physical_no_overlap": split.passes_physical_no_overlap,
                **fingerprints,
                "reason": split.reason,
            }
        )
    return rows
