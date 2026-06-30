from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import _iwm_bootstrap  # noqa: F401

from industrial_world_model.sensor.datasets import load_or_synthetic_sensor_windows
from industrial_world_model.sensor.eval import sensor_scores_report
from industrial_world_model.sensor.lejepa_sensor import windows_to_engineered_matrix


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate sensor engineered baseline for IWM.")
    parser.add_argument("--config", default="configs/industrial_world_model/lejepa_sensor_cnc.yaml")
    parser.add_argument("--out-root", default="outputs/industrial_world_model/sensor_lejepa")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    out = Path(args.out_root)
    out.mkdir(parents=True, exist_ok=True)
    x, y = load_or_synthetic_sensor_windows("data/raw/sensor/cnc_milling")
    feats = windows_to_engineered_matrix(x)
    xtr, xte, ytr, yte = train_test_split(feats, y, test_size=0.4, random_state=42, stratify=y if len(np.unique(y)) > 1 else None)
    clf = RandomForestClassifier(n_estimators=50, random_state=42, class_weight="balanced").fit(xtr, ytr)
    scores = clf.predict_proba(xte)[:, 1]
    row = {"model": "sensor_engineered_random_forest", **sensor_scores_report(yte, scores), "notes": "synthetic fallback if CNC CSV not found"}
    pd.DataFrame([row]).to_csv(out / "results.csv", index=False)
    pd.DataFrame([row]).to_csv(out / "results_mean_std.csv", index=False)
    pd.DataFrame([{"delta_AUPRC_vs_engineered": 0.0, "notes": "baseline row"}]).to_csv(out / "deltas_vs_engineered.csv", index=False)
    (out / "report.md").write_text("# Sensor LeJEPA Report\n\nEngineered sensor baseline evaluated; LeJEPA deltas require pretrained embeddings.\n", encoding="utf-8")
    print(f"Sensor evaluation written to {out}")


if __name__ == "__main__":
    main()
