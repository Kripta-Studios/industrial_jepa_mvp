from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import _bootstrap  # noqa: F401
from common.paths import ensure_dir
from common.reports import markdown_table, write_markdown_report


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _rows(df: pd.DataFrame, names: list[str], protocol: str | None = None) -> pd.DataFrame:
    if df.empty or "model_name" not in df:
        return pd.DataFrame()
    out = df[df["model_name"].isin(names)].copy()
    if protocol is not None and "protocol" in out:
        out = out[out["protocol"].eq(protocol)]
    keep = [
        c
        for c in [
            "protocol",
            "split_name",
            "model_name",
            "AUPRC",
            "AUROC",
            "precision_at_10pct",
            "recall_at_10pct",
            "delta_AUPRC_vs_metadata_only",
            "delta_Precision@10_vs_metadata_only",
            "delta_Recall@10_vs_metadata_only",
            "AUPRC_mean",
            "AUPRC_std",
            "delta_AUPRC_vs_metadata_only_mean",
            "delta_AUPRC_vs_metadata_only_std",
        ]
        if c in out.columns
    ]
    sort_col = "AUPRC_mean" if "AUPRC_mean" in out else "AUPRC" if "AUPRC" in out else None
    if sort_col:
        out = out.sort_values(sort_col, ascending=False)
    return out[keep]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--focused-root", default="outputs/sensor_jepa/incremental_value_h3_k10_3seed")
    parser.add_argument("--gbt-root", default="outputs/sensor_jepa/incremental_value_benchmark")
    parser.add_argument("--hard-root", default="outputs/sensor_jepa/hard_generalization_extended")
    parser.add_argument("--dense-root", default="outputs/sensor_jepa/dense_sensor_jepa_cnc")
    parser.add_argument("--out", default="outputs/sensor_jepa/incremental_value_next_step_report.md")
    args = parser.parse_args()

    focused = _read(Path(args.focused_root) / "incremental_results.csv")
    focused_mean = _read(Path(args.focused_root) / "results_mean_std.csv")
    gbt = _read(Path(args.gbt_root) / "gbt_audit_results.csv")
    hard = _read(Path(args.hard_root) / "results.csv")
    dense = _read(Path(args.dense_root) / "incremental_results.csv")

    core = [
        "metadata_only",
        "metadata_plus_current_z",
        "metadata_plus_current_z_plus_predicted_future_z",
        "metadata_plus_predicted_future_z",
        "metadata_plus_world_model_score",
    ]
    focused_table = _rows(focused_mean.rename(columns={c: c for c in focused_mean.columns}), core, protocol="operational")
    hard_table = _rows(
        hard,
        [
            "metadata_only",
            "sensor_raw_only",
            "current_z_only",
            "predicted_future_z_only",
            "metadata_plus_current_z",
            "metadata_plus_current_z_plus_predicted_future_z",
        ],
    )
    dense_table = _rows(
        dense,
        [
            "dense_sensor_topk_surprise",
            "dense_sensor_negative_topk_surprise",
            "metadata_only",
            "metadata_plus_surprise",
            "metadata_plus_negative_surprise",
            "metadata_plus_dense_embedding_pool",
        ],
    )

    def answer_delta(name: str) -> str:
        if focused_mean.empty:
            return "pending"
        row = focused_mean[(focused_mean["protocol"].eq("operational")) & (focused_mean["model_name"].eq(name))]
        if row.empty:
            return "pending"
        mean = row.iloc[0].get("delta_AUPRC_vs_metadata_only_mean")
        std = row.iloc[0].get("delta_AUPRC_vs_metadata_only_std")
        return f"{mean:+.4f} +/- {std:.4f}" if pd.notna(std) else f"{mean:+.4f}"

    report_path = Path(args.out)
    ensure_dir(report_path.parent)
    write_markdown_report(
        report_path,
        "Sensor Incremental Value Next Step Report",
        {
            "Current Z Delta Stability": answer_delta("metadata_plus_current_z"),
            "Current Z Plus Predicted Future Delta Stability": answer_delta("metadata_plus_current_z_plus_predicted_future_z"),
            "Focused 3-Seed Operational Rows": markdown_table(focused_table.to_dict("records")) if len(focused_table) else "pending",
            "GBT Audit": markdown_table(gbt.to_dict("records")) if len(gbt) else "pending",
            "Hard Generalization Extended": markdown_table(hard_table.head(40).to_dict("records")) if len(hard_table) else "pending",
            "DenseSensor Surprise Sanity": markdown_table(dense_table.to_dict("records")) if len(dense_table) else "pending",
            "Claim": (
                "Honest claim remains: Sensor-JEPA current embeddings show incremental value over metadata/cycle in selected protocols, "
                "pending multi-seed stability and audited tree baselines. No SOTA claim."
            ),
        },
    )
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
