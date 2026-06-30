from __future__ import annotations

from pathlib import Path

import pandas as pd


def main() -> None:
    rows = []
    for root in [Path("outputs/sensor_jepa"), Path("outputs/visual_jepa")]:
        for path in root.rglob("*.csv"):
            if path.name.endswith("history.csv"):
                continue
            try:
                df = pd.read_csv(path)
                df["source_file"] = str(path)
                rows.append(df)
            except Exception:
                pass
    out = Path("outputs/reports/experiment_comparison.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        pd.concat(rows, ignore_index=True).to_csv(out, index=False)
    else:
        pd.DataFrame().to_csv(out, index=False)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

