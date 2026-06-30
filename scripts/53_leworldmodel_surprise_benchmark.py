from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Build LeWorldModel surprise benchmark table.")
    parser.add_argument("--config", default="configs/industrial_world_model/leworldmodel_cnc.yaml")
    parser.add_argument("--out-root", default="outputs/industrial_world_model/leworldmodel")
    args = parser.parse_args()
    out = Path(args.out_root)
    out.mkdir(parents=True, exist_ok=True)
    src = out / "surprise_results.csv"
    if src.exists():
        df = pd.read_csv(src)
    else:
        df = pd.DataFrame([{"dataset": "synthetic", "surprise_AUROC": None, "surprise_AUPRC": None, "notes": "run pretrain first"}])
    df.to_csv(out / "results_mean_std.csv", index=False)
    print(f"Surprise benchmark written to {out / 'results_mean_std.csv'}")


if __name__ == "__main__":
    main()
