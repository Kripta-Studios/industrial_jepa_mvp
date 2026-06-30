from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import _bootstrap  # noqa: F401
from common.paths import ensure_dir
from common.reports import markdown_table, write_markdown_report


def _read(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _top(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    if df.empty:
        return df
    sort_col = "AUPRC_mean" if "AUPRC_mean" in df.columns else "AUPRC" if "AUPRC" in df.columns else None
    return df.sort_values(sort_col, ascending=False).head(n) if sort_col else df.head(n)


def _verdict_from_delta(df: pd.DataFrame, model: str, threshold: float = 0.05, protocol: str | None = None) -> str:
    if df.empty or "model_name" not in df.columns:
        return "pending"
    row = df[df["model_name"].eq(model)]
    if protocol is not None and "protocol" in row.columns:
        row = row[row["protocol"].eq(protocol)]
    if row.empty:
        return "pending"
    col = "delta_AUPRC_vs_metadata_only_mean" if "delta_AUPRC_vs_metadata_only_mean" in row.columns else "delta_AUPRC_vs_metadata_only"
    if col not in row.columns:
        return "pending"
    value = float(row[col].mean())
    std_col = "delta_AUPRC_vs_metadata_only_std"
    std = float(row[std_col].mean()) if std_col in row.columns else 0.0
    if value >= threshold and std <= abs(value):
        return f"parcial/fuerte: delta {value:+.4f}"
    if value > 0:
        return f"parcial: delta {value:+.4f}, varianza alta"
    return f"debil: delta {value:+.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs/sensor_jepa/sensor_evidence_consolidated_report.md")
    parser.add_argument("--operational", default="outputs/sensor_jepa/incremental_value_h3_k10_3seed/results_mean_std.csv")
    parser.add_argument("--no-cycle", default="outputs/sensor_jepa/no_cycle_evidence/results_mean_std.csv")
    parser.add_argument("--hard", default="outputs/sensor_jepa/hard_generalization_evidence/results_mean_std.csv")
    parser.add_argument("--raw-jepa", default="outputs/sensor_jepa/sensor_vs_jepa_value/no_cycle_summary.csv")
    parser.add_argument("--hard-raw-jepa", default="outputs/sensor_jepa/sensor_vs_jepa_value/hard_generalization_summary.csv")
    parser.add_argument("--dense", default="outputs/sensor_jepa/dense_sensor_jepa_cnc/incremental_results.csv")
    args = parser.parse_args()

    operational = _read(args.operational)
    no_cycle = _read(args.no_cycle)
    hard = _read(args.hard)
    raw_jepa = _read(args.raw_jepa)
    hard_raw_jepa = _read(args.hard_raw_jepa)
    dense = _read(args.dense)

    no_cycle_core = no_cycle[no_cycle["model_name"].isin(["metadata_only_no_cycle", "sensor_raw_only", "current_z_only", "predicted_future_z_only", "metadata_plus_current_z_plus_predicted_future_z"])] if not no_cycle.empty else pd.DataFrame()
    hard_core = hard[hard["model_name"].isin(["metadata_only", "sensor_raw_only", "current_z_only", "predicted_future_z_only", "metadata_plus_current_z_plus_predicted_future_z"])] if not hard.empty else pd.DataFrame()
    dense_core = dense[dense["model_name"].astype(str).str.contains("dense", regex=False) | dense["model_name"].eq("metadata_only")] if not dense.empty else pd.DataFrame()

    out = Path(args.out)
    ensure_dir(out.parent)
    write_markdown_report(
        out,
        "Sensor Evidence Consolidated Report",
        {
            "Operational Evidence": _verdict_from_delta(operational, "metadata_plus_current_z_plus_predicted_future_z", protocol="operational"),
            "No-Cycle Evidence": _verdict_from_delta(no_cycle, "metadata_plus_current_z_plus_predicted_future_z"),
            "Hard-Generalization Evidence": "partial: inspect split-level table; value is split-dependent.",
            "Raw Sensor vs JEPA": "partial: raw-vs-JEPA deltas are reported separately; JEPA must beat or complement raw within same seed/h/K.",
            "Predicted Future Z": _verdict_from_delta(no_cycle, "predicted_future_z_only"),
            "DenseSensorJEPA": "no current evidence unless dense diagnostic rows beat metadata/current_z/sensor_raw.",
            "SOTA Candidate": "No.",
            "Commercial Product Claim": "A metadata/cycle risk scorer with optional sensor/JEPA features is honest. Sensor features are most defensible in no-cycle or hard-generalization settings.",
            "Operational Mean/Std": markdown_table(_top(operational).to_dict("records")) if len(operational) else "pending",
            "No-Cycle Mean/Std": markdown_table(_top(no_cycle_core, 30).to_dict("records")) if len(no_cycle_core) else "pending",
            "Hard Mean/Std": markdown_table(_top(hard_core, 40).to_dict("records")) if len(hard_core) else "pending",
            "No-Cycle Raw-vs-JEPA": markdown_table(_top(raw_jepa, 30).to_dict("records")) if len(raw_jepa) else "pending",
            "Hard Raw-vs-JEPA": markdown_table(_top(hard_raw_jepa, 30).to_dict("records")) if len(hard_raw_jepa) else "pending",
            "Dense Diagnostic": markdown_table(_top(dense_core, 30).to_dict("records")) if len(dense_core) else "pending",
        },
    )
    print(f"report: {out}")


if __name__ == "__main__":
    main()
