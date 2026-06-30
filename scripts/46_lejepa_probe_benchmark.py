from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import _iwm_bootstrap  # noqa: F401


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LeJEPA frozen probe benchmark placeholder.")
    parser.add_argument("--config", default="configs/industrial_world_model/lejepa_visual_mvtec.yaml")
    parser.add_argument("--out-root", default="outputs/industrial_world_model/lejepa_visual")
    args = parser.parse_args()
    out = Path(args.out_root)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"dataset": "pending", "probe_AUROC": None, "probe_AUPRC": None, "notes": "requires labels/features"}]).to_csv(out / "probe_results.csv", index=False)
    print(f"Probe results written to {out / 'probe_results.csv'}")


if __name__ == "__main__":
    main()
