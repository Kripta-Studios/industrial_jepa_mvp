from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import _iwm_bootstrap  # noqa: F401

from industrial_world_model.hierarchy.aggregation import group_risk_table, top_alerts
from industrial_world_model.hierarchy.reports import write_hierarchy_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run hierarchical IWM aggregation benchmark.")
    parser.add_argument("--out-root", default="outputs/industrial_world_model/hierarchy")
    args = parser.parse_args()
    rng = np.random.default_rng(42)
    df = pd.DataFrame({"item_id": [f"window_{i:03d}" for i in range(50)], "cycle": [f"cycle_{i//5:02d}" for i in range(50)], "risk_score": rng.random(50)})
    group = group_risk_table(df, "cycle", "risk_score")
    out = Path(args.out_root)
    write_hierarchy_report(out, group)
    top_alerts(df).to_csv(out / "window_top_alerts.csv", index=False)
    print(f"Hierarchy results written to {out}")


if __name__ == "__main__":
    main()
