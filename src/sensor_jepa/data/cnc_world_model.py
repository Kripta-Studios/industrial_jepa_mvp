from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .cnc_milling import ACTION_COLUMNS, META_COLUMNS, load_feature_table, split_tools
from .normalization import Standardizer
from .windowing import sliding_windows


@dataclass
class TransitionBundle:
    x_train: np.ndarray
    a_train: np.ndarray
    x_next_train: np.ndarray
    y_failure_train: np.ndarray
    x_val: np.ndarray
    a_val: np.ndarray
    x_next_val: np.ndarray
    y_failure_val: np.ndarray
    x_test: np.ndarray
    a_test: np.ndarray
    x_next_test: np.ndarray
    y_failure_test: np.ndarray
    train_meta: pd.DataFrame
    val_meta: pd.DataFrame
    test_meta: pd.DataFrame
    feature_names: list[str]
    action_names: list[str]
    standardizer: Standardizer
    action_standardizer: Standardizer

    @property
    def input_channels(self) -> int:
        return int(self.x_train.shape[-1])

    @property
    def action_dim(self) -> int:
        return int(self.a_train.shape[-1])


def build_cnc_transition_bundle(
    raw_root: str | Path,
    window_length: int = 8,
    stride: int = 1,
    forecast_horizon: int = 1,
    failure_horizon_cycles: int = 10,
    val_fraction: float = 0.15,
    test_fraction: float = 0.20,
    seed: int = 42,
) -> TransitionBundle:
    df = load_feature_table(raw_root)
    feature_names = [c for c in df.columns if c not in META_COLUMNS and not c.startswith("wear_class")]
    feature_names = [c for c in feature_names if pd.api.types.is_numeric_dtype(df[c])]
    action_names = [c for c in ACTION_COLUMNS if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if not action_names:
        raise ValueError("No numeric CNC action/context columns were found.")
    train_tools, val_tools, test_tools = split_tools(
        df["ToolIndex"].dropna().unique(),
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )
    parts: dict[str, list[np.ndarray]] = {k: [] for k in ["x", "a", "x_next", "y_failure"]}
    meta_parts: dict[str, list[pd.DataFrame]] = {"train": [], "val": [], "test": []}
    split_arrays = {
        "train": {k: [] for k in ["x", "a", "x_next", "y_failure"]},
        "val": {k: [] for k in ["x", "a", "x_next", "y_failure"]},
        "test": {k: [] for k in ["x", "a", "x_next", "y_failure"]},
    }

    for tool_id, g in df.groupby("ToolIndex"):
        g = g.sort_values(["NumberOfCycle", "FileName"]).reset_index(drop=True)
        values = g[feature_names].to_numpy(dtype=np.float32)
        windows, last_idx = sliding_windows(values, window_length, stride)
        if len(windows) <= forecast_horizon:
            continue
        current = windows[:-forecast_horizon]
        target = windows[forecast_horizon:]
        current_idx = last_idx[:-forecast_horizon]
        target_idx = last_idx[forecast_horizon:]
        actions = g.loc[current_idx, action_names].to_numpy(dtype=np.float32)
        actions = np.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0)
        y_failure = (g.loc[target_idx, "CycleToFailure"].to_numpy(dtype=float) <= failure_horizon_cycles).astype(np.int64)
        meta_cols = [
            "FileName",
            "NumberOfCycle",
            "ToolIndex",
            "CycleToFailure",
            "CycleToFailureNormalized",
            "wear_class_name",
        ] + action_names
        meta = g.loc[target_idx, meta_cols].copy()
        meta["source_cycle"] = g.loc[current_idx, "NumberOfCycle"].to_numpy()
        meta["forecast_horizon"] = forecast_horizon
        meta["failure_horizon_cycles"] = failure_horizon_cycles
        split = "train" if int(tool_id) in train_tools else "val" if int(tool_id) in val_tools else "test"
        split_arrays[split]["x"].append(current)
        split_arrays[split]["a"].append(actions)
        split_arrays[split]["x_next"].append(target)
        split_arrays[split]["y_failure"].append(y_failure)
        meta_parts[split].append(meta)

    def cat(split: str, key: str, shape_tail: tuple[int, ...], dtype=np.float32) -> np.ndarray:
        if not split_arrays[split][key]:
            return np.empty((0, *shape_tail), dtype=dtype)
        return np.concatenate(split_arrays[split][key], axis=0).astype(dtype)

    def cat_meta(split: str) -> pd.DataFrame:
        if not meta_parts[split]:
            return pd.DataFrame()
        return pd.concat(meta_parts[split], ignore_index=True)

    x_train = cat("train", "x", (window_length, len(feature_names)))
    x_next_train = cat("train", "x_next", (window_length, len(feature_names)))
    a_train = cat("train", "a", (len(action_names),))
    y_failure_train = cat("train", "y_failure", (), dtype=np.int64)

    x_val = cat("val", "x", (window_length, len(feature_names)))
    x_next_val = cat("val", "x_next", (window_length, len(feature_names)))
    a_val = cat("val", "a", (len(action_names),))
    y_failure_val = cat("val", "y_failure", (), dtype=np.int64)

    x_test = cat("test", "x", (window_length, len(feature_names)))
    x_next_test = cat("test", "x_next", (window_length, len(feature_names)))
    a_test = cat("test", "a", (len(action_names),))
    y_failure_test = cat("test", "y_failure", (), dtype=np.int64)

    standardizer = Standardizer.fit(np.concatenate([x_train, x_next_train], axis=0))
    action_standardizer = Standardizer.fit(a_train[:, None, :])

    return TransitionBundle(
        x_train=standardizer.transform(x_train),
        a_train=action_standardizer.transform(a_train[:, None, :])[:, 0],
        x_next_train=standardizer.transform(x_next_train),
        y_failure_train=y_failure_train,
        x_val=standardizer.transform(x_val),
        a_val=action_standardizer.transform(a_val[:, None, :])[:, 0],
        x_next_val=standardizer.transform(x_next_val),
        y_failure_val=y_failure_val,
        x_test=standardizer.transform(x_test),
        a_test=action_standardizer.transform(a_test[:, None, :])[:, 0],
        x_next_test=standardizer.transform(x_next_test),
        y_failure_test=y_failure_test,
        train_meta=cat_meta("train"),
        val_meta=cat_meta("val"),
        test_meta=cat_meta("test"),
        feature_names=feature_names,
        action_names=action_names,
        standardizer=standardizer,
        action_standardizer=action_standardizer,
    )


def prepare_transition_from_config(cfg: dict[str, Any]) -> TransitionBundle:
    data_cfg = cfg["data"]
    world_cfg = cfg.get("world_model", {})
    return build_cnc_transition_bundle(
        raw_root=data_cfg["raw_root"],
        window_length=int(data_cfg.get("window_length", 8)),
        stride=int(data_cfg.get("stride", 1)),
        forecast_horizon=int(world_cfg.get("forecast_horizon", 1)),
        failure_horizon_cycles=int(world_cfg.get("failure_horizon_cycles", 10)),
        val_fraction=float(data_cfg.get("val_fraction", 0.15)),
        test_fraction=float(data_cfg.get("test_fraction", 0.20)),
        seed=int(cfg.get("seed", 42)),
    )

