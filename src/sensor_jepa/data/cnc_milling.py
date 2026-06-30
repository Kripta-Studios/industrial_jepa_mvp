from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .normalization import Standardizer
from .windowing import sliding_windows


META_COLUMNS = {
    "FileName",
    "NumberOfCycle",
    "SampleIndex",
    "TollIndex",
    "ToolIndex",
    "MillingToolType",
    "ADOC",
    "RDOC",
    "HardnessMean",
    "ToolHolderLength",
    "ToolRotation",
    "FeedRate",
    "ToolDiameter",
    "CycleToFailure",
    "CycleToFailureNormalized",
}

ACTION_COLUMNS = [
    "MillingToolType",
    "ADOC",
    "RDOC",
    "HardnessMean",
    "ToolHolderLength",
    "ToolRotation",
    "FeedRate",
    "ToolDiameter",
]


@dataclass
class SensorBundle:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    train_meta: pd.DataFrame
    val_meta: pd.DataFrame
    test_meta: pd.DataFrame
    feature_names: list[str]
    class_names: list[str]
    standardizer: Standardizer

    @property
    def input_channels(self) -> int:
        return int(self.x_train.shape[-1])


def _coerce_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.replace(",", ".", regex=False)
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_feature_table(raw_root: str | Path) -> pd.DataFrame:
    raw_root = Path(raw_root)
    path = raw_root / "FeatureAndMetadata_Milling.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing CNC feature file: {path}")
    df = pd.read_csv(path, sep=";", header=1, decimal=",", low_memory=False)
    if "TollIndex" in df.columns and "ToolIndex" not in df.columns:
        df["ToolIndex"] = df["TollIndex"]
    metadata_path = raw_root / "metadata.xlsx"
    if metadata_path.exists():
        meta = pd.read_excel(metadata_path)
        meta = meta.rename(
            columns={
                "ExperimentIndex": "FileName",
                "ToolIndex": "ToolIndex",
                "ADOC [mm]": "ADOC",
                "RDOC [mm]": "RDOC",
                "HardnessMean [HRC]": "HardnessMean",
                "ToolHolderLength [mm]": "ToolHolderLength",
                "ToolRotation [rpm]": "ToolRotation",
                "FeedRate [mm/min]": "FeedRate",
                "ToolDiameter [mm]": "ToolDiameter",
            }
        )
        keep = ["FileName"] + [c for c in ACTION_COLUMNS + ["ToolIndex"] if c in meta.columns]
        meta = meta[keep].drop_duplicates("FileName")
        add_cols = [c for c in keep if c != "FileName" and (c not in df.columns or df[c].isna().all())]
        if add_cols:
            df = df.merge(meta[["FileName"] + add_cols], on="FileName", how="left")
    feature_cols = [c for c in df.columns if c not in META_COLUMNS]
    numeric_cols = feature_cols + [
        c
        for c in [
            "NumberOfCycle",
            "SampleIndex",
            "ToolIndex",
            "TollIndex",
            "MillingToolType",
            "ADOC",
            "RDOC",
            "HardnessMean",
            "ToolHolderLength",
            "ToolRotation",
            "FeedRate",
            "ToolDiameter",
            "CycleToFailure",
            "CycleToFailureNormalized",
        ]
        if c in df.columns
    ]
    df = _coerce_numeric(df, numeric_cols)
    df = df.copy()
    df["wear_class"] = pd.cut(
        df["CycleToFailureNormalized"],
        bins=[-0.001, 0.33, 0.66, 1.001],
        labels=[2, 1, 0],
    ).astype(int)
    df["wear_class_name"] = df["wear_class"].map({0: "Healthy", 1: "Moderate", 2: "Worn"})
    return df


def split_tools(
    tools: np.ndarray,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[set[int], set[int], set[int]]:
    tools = np.array(sorted(set(int(t) for t in tools)))
    rng = np.random.default_rng(seed)
    rng.shuffle(tools)
    n = len(tools)
    n_test = max(1, int(round(n * test_fraction)))
    n_val = max(1, int(round(n * val_fraction)))
    test = set(tools[:n_test].tolist())
    val = set(tools[n_test : n_test + n_val].tolist())
    train = set(tools[n_test + n_val :].tolist())
    if not train:
        train = set(tools[n_test + n_val - 1 : n_test + n_val].tolist())
    return train, val, test


def build_cnc_windows(
    raw_root: str | Path,
    window_length: int = 8,
    stride: int = 1,
    val_fraction: float = 0.15,
    test_fraction: float = 0.20,
    seed: int = 42,
    max_windows: int | None = None,
) -> SensorBundle:
    df = load_feature_table(raw_root)
    feature_names = [c for c in df.columns if c not in META_COLUMNS and not c.startswith("wear_class")]
    feature_names = [c for c in feature_names if pd.api.types.is_numeric_dtype(df[c])]
    train_tools, val_tools, test_tools = split_tools(
        df["ToolIndex"].dropna().unique(),
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )

    parts: dict[str, list[np.ndarray]] = {"train": [], "val": [], "test": []}
    labels: dict[str, list[np.ndarray]] = {"train": [], "val": [], "test": []}
    metas: dict[str, list[pd.DataFrame]] = {"train": [], "val": [], "test": []}

    for tool_id, g in df.groupby("ToolIndex"):
        g = g.sort_values(["NumberOfCycle", "FileName"]).reset_index(drop=True)
        x = g[feature_names].to_numpy(dtype=np.float32)
        windows, idx = sliding_windows(x, window_length, stride)
        if len(windows) == 0:
            continue
        y = g.loc[idx, "wear_class"].to_numpy(dtype=np.int64)
        meta_cols = [
            "FileName",
            "NumberOfCycle",
            "ToolIndex",
            "CycleToFailure",
            "CycleToFailureNormalized",
            "wear_class_name",
        ] + [c for c in ACTION_COLUMNS if c in g.columns]
        m = g.loc[idx, meta_cols].copy()
        split = "train" if int(tool_id) in train_tools else "val" if int(tool_id) in val_tools else "test"
        parts[split].append(windows)
        labels[split].append(y)
        metas[split].append(m)

    def cat_x(name: str) -> np.ndarray:
        if not parts[name]:
            return np.empty((0, window_length, len(feature_names)), dtype=np.float32)
        return np.concatenate(parts[name], axis=0)

    def cat_y(name: str) -> np.ndarray:
        if not labels[name]:
            return np.empty((0,), dtype=np.int64)
        return np.concatenate(labels[name], axis=0)

    def cat_m(name: str) -> pd.DataFrame:
        if not metas[name]:
            return pd.DataFrame()
        return pd.concat(metas[name], axis=0, ignore_index=True)

    x_train, y_train = cat_x("train"), cat_y("train")
    x_val, y_val = cat_x("val"), cat_y("val")
    x_test, y_test = cat_x("test"), cat_y("test")
    train_meta, val_meta, test_meta = cat_m("train"), cat_m("val"), cat_m("test")

    if max_windows is not None:
        x_train, y_train, train_meta = x_train[:max_windows], y_train[:max_windows], train_meta.iloc[:max_windows].reset_index(drop=True)

    standardizer = Standardizer.fit(x_train)
    x_train = standardizer.transform(x_train)
    x_val = standardizer.transform(x_val)
    x_test = standardizer.transform(x_test)
    return SensorBundle(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        train_meta=train_meta,
        val_meta=val_meta,
        test_meta=test_meta,
        feature_names=feature_names,
        class_names=["Healthy", "Moderate", "Worn"],
        standardizer=standardizer,
    )


def save_bundle(bundle: SensorBundle, processed_path: str | Path, manifest_path: str | Path | None = None) -> Path:
    processed_path = Path(processed_path)
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        processed_path,
        x_train=bundle.x_train,
        y_train=bundle.y_train,
        x_val=bundle.x_val,
        y_val=bundle.y_val,
        x_test=bundle.x_test,
        y_test=bundle.y_test,
        feature_names=np.array(bundle.feature_names, dtype=object),
        class_names=np.array(bundle.class_names, dtype=object),
        mean=bundle.standardizer.mean,
        std=bundle.standardizer.std,
    )
    if manifest_path is not None:
        manifest_path = Path(manifest_path)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        all_meta = []
        for split, meta in [("train", bundle.train_meta), ("val", bundle.val_meta), ("test", bundle.test_meta)]:
            m = meta.copy()
            m["split"] = split
            all_meta.append(m)
        pd.concat(all_meta, ignore_index=True).to_csv(manifest_path, index=False)
    return processed_path


def load_processed(path: str | Path) -> SensorBundle:
    path = Path(path)
    data = np.load(path, allow_pickle=True)
    standardizer = Standardizer(mean=data["mean"], std=data["std"])
    return SensorBundle(
        x_train=data["x_train"],
        y_train=data["y_train"],
        x_val=data["x_val"],
        y_val=data["y_val"],
        x_test=data["x_test"],
        y_test=data["y_test"],
        train_meta=pd.DataFrame(),
        val_meta=pd.DataFrame(),
        test_meta=pd.DataFrame(),
        feature_names=list(data["feature_names"]),
        class_names=list(data["class_names"]),
        standardizer=standardizer,
    )


def prepare_from_config(cfg: dict[str, Any], force: bool = False) -> SensorBundle:
    data_cfg = cfg["data"]
    processed_path = Path(data_cfg["processed_path"])
    if processed_path.exists() and not force:
        return load_processed(processed_path)
    bundle = build_cnc_windows(
        raw_root=data_cfg["raw_root"],
        window_length=int(data_cfg.get("window_length", 8)),
        stride=int(data_cfg.get("stride", 1)),
        val_fraction=float(data_cfg.get("val_fraction", 0.15)),
        test_fraction=float(data_cfg.get("test_fraction", 0.20)),
        seed=int(cfg.get("seed", 42)),
        max_windows=data_cfg.get("max_windows"),
    )
    save_bundle(bundle, processed_path, data_cfg.get("manifest_path"))
    return bundle


def read_raw_csv_sample(path: str | Path, nrows: int = 2048) -> pd.DataFrame:
    return pd.read_csv(path, nrows=nrows)
