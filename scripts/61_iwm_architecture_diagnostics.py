from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _score(row: pd.Series, metric: str) -> float | None:
    value = row.get(metric)
    if pd.isna(value):
        return None
    return float(value)


def _append_visual_foundation(rows: list[dict], root: Path) -> None:
    df = _read_csv(root / "visual_foundation" / "results.csv")
    if df is None or df.empty or "image_AUPRC" not in df:
        rows.append(
            {
                "component": "visual_foundation",
                "comparison": "DINOv2/ResNet PatchCore/PaDiM vs pixel baseline",
                "status": "missing",
                "metric": "image_AUPRC",
                "baseline": "pixel_stat_baseline",
                "candidate": "dense_feature_patchcore_padim",
                "baseline_score": None,
                "candidate_score": None,
                "delta": None,
                "claim": "pending",
                "notes": "Run scripts/41_visual_foundation_benchmark.py first.",
            }
        )
        return
    pixel = df[df["model"].eq("pixel_stat_baseline")]
    candidates = df[~df["model"].eq("pixel_stat_baseline")].copy()
    if pixel.empty or candidates.empty:
        return
    best = candidates.sort_values("image_AUPRC", ascending=False).iloc[0]
    base = pixel.iloc[0]
    base_score = _score(base, "image_AUPRC")
    cand_score = _score(best, "image_AUPRC")
    delta = None if base_score is None or cand_score is None else cand_score - base_score
    rows.append(
        {
            "component": "visual_foundation",
            "comparison": "best dense-feature visual scorer vs pixel baseline",
            "status": "ok",
            "metric": "image_AUPRC",
            "baseline": "pixel_stat_baseline",
            "candidate": best["model"],
            "baseline_score": base_score,
            "candidate_score": cand_score,
            "delta": delta,
            "claim": "strong local baseline improvement" if delta is not None and delta > 0.05 else "no meaningful local delta",
            "notes": "Product-relevant only after multi-category validation; current quick runs are not literature-level evidence.",
        }
    )


def _append_dense_visual(rows: list[dict], project_root: Path) -> None:
    df = _read_csv(project_root / "outputs" / "visual_jepa" / "dense_benchmark" / "results.csv")
    if df is None or df.empty or "image_AUPRC" not in df:
        rows.append(
            {
                "component": "dense_visual_jepa",
                "comparison": "DenseVisualJEPA vs ResNet/DINO PatchCore/PaDiM",
                "status": "missing",
                "metric": "image_AUPRC",
                "baseline": "strong_frozen_features",
                "candidate": "dense_visual_jepa",
                "baseline_score": None,
                "candidate_score": None,
                "delta": None,
                "claim": "pending",
                "notes": "Run scripts/26_benchmark_dense_visual_jepa.py first.",
            }
        )
        return
    dense = df[df["model_name"].astype(str).str.contains("dense_visual_jepa", na=False)]
    strong = df[df["model_name"].astype(str).str.contains("resnet|dino", case=False, na=False)]
    if dense.empty or strong.empty:
        return
    dense_best = dense.sort_values("image_AUPRC", ascending=False).iloc[0]
    strong_best = strong.sort_values("image_AUPRC", ascending=False).iloc[0]
    dense_score = _score(dense_best, "image_AUPRC")
    strong_score = _score(strong_best, "image_AUPRC")
    delta = None if dense_score is None or strong_score is None else dense_score - strong_score
    rows.append(
        {
            "component": "dense_visual_jepa",
            "comparison": "best DenseVisualJEPA scorer vs best ResNet/DINO scorer",
            "status": "ok",
            "metric": "image_AUPRC",
            "baseline": strong_best["model_name"],
            "candidate": dense_best["model_name"],
            "baseline_score": strong_score,
            "candidate_score": dense_score,
            "delta": delta,
            "claim": "research only; below strong frozen-feature baseline" if delta is not None and delta < -0.02 else "competitive locally, needs broader validation",
            "notes": "Use kNN/PaDiM over dense features; direct latent error is not product-ready.",
        }
    )


def _append_sensor_value(rows: list[dict], project_root: Path) -> None:
    path = project_root / "outputs" / "sensor_jepa" / "jepa_vs_engineered_value" / "results_mean_std.csv"
    df = _read_csv(path)
    if df is None or df.empty:
        rows.append(
            {
                "component": "sensor_jepa",
                "comparison": "JEPA/future features vs engineered sensor features",
                "status": "missing",
                "metric": "AUPRC",
                "baseline": "sensor_engineered_only",
                "candidate": "jepa_features",
                "baseline_score": None,
                "candidate_score": None,
                "delta": None,
                "claim": "pending",
                "notes": "Run the jepa_vs_engineered benchmark first.",
            }
        )
        return
    candidate = df[df["model_name"].astype(str).str.contains("current_z|predicted_future_z", na=False)].copy()
    if candidate.empty or "delta_AUPRC_vs_sensor_engineered_mean" not in candidate:
        return
    best = candidate.sort_values("delta_AUPRC_vs_sensor_engineered_mean", ascending=False).iloc[0]
    delta = _score(best, "delta_AUPRC_vs_sensor_engineered_mean")
    rows.append(
        {
            "component": "sensor_jepa",
            "comparison": "best JEPA/future feature delta vs sensor_engineered_only",
            "status": "ok",
            "metric": "delta_AUPRC_vs_sensor_engineered",
            "baseline": "sensor_engineered_only",
            "candidate": best["model_name"],
            "baseline_score": None,
            "candidate_score": None,
            "delta": delta,
            "claim": "partial incremental value" if delta is not None and delta > 0.02 else "not robustly better than engineered features",
            "notes": f"Best protocol/split: {best.get('protocol')}/{best.get('split_name')}. Do not claim replacement of engineered features.",
        }
    )


def _append_training_stability(rows: list[dict], root: Path) -> None:
    lejepa = _read_csv(root / "lejepa_visual" / "pretrain_log.csv")
    if lejepa is not None and not lejepa.empty:
        last = lejepa.iloc[-1]
        rows.append(
            {
                "component": "lejepa_sigreg",
                "comparison": "training stability and collapse diagnostics",
                "status": "ok",
                "metric": "collapse_flag",
                "baseline": "n/a",
                "candidate": "lejepa_sigreg",
                "baseline_score": None,
                "candidate_score": bool(last.get("collapse_flag", True)),
                "delta": None,
                "claim": "stable smoke training" if not bool(last.get("collapse_flag", True)) else "collapse risk",
                "notes": "Stability is not downstream product evidence; evaluate via PatchCore/kNN/probes.",
            }
        )
    wm = _read_csv(root / "leworldmodel" / "results.csv")
    if wm is not None and not wm.empty:
        last = wm.iloc[-1]
        rows.append(
            {
                "component": "leworldmodel",
                "comparison": "synthetic latent prediction and surprise",
                "status": "ok",
                "metric": "prediction_loss",
                "baseline": "synthetic_target",
                "candidate": "leworldmodel_sigreg",
                "baseline_score": None,
                "candidate_score": _score(last, "prediction_loss"),
                "delta": None,
                "claim": "synthetic smoke only",
                "notes": "Needs real temporal process data/actions before any predictive-quality claim.",
            }
        )


def _write_report(df: pd.DataFrame, out_dir: Path) -> None:
    if df.empty:
        table = "No diagnostics were available."
    else:
        headers = list(df.columns)
        rows = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
        for _, row in df.iterrows():
            rows.append("| " + " | ".join(str(row.get(col, "")) for col in headers) + " |")
        table = "\n".join(rows)
    lines = [
        "# IWM Architecture Upgrade Diagnostics",
        "",
        "This report answers whether the paper-inspired changes improve local industrial baselines.",
        "It does not assert literature SOTA unless broad external comparisons exist.",
        "",
        "## Summary Table",
        "",
        table,
        "",
        "## Decision",
        "",
        "- Product core: use dense frozen visual features plus PatchCore/PaDiM when they beat simple baselines.",
        "- DenseVisualJEPA: keep as research/adaptation until it beats or complements ResNet/DINO baselines.",
        "- Sensor JEPA: keep optional unless it adds stable delta over engineered sensor features.",
        "- LeWorldModel: use as temporal-surprise research path; real evidence requires sequence/action data.",
        "- Literature SOTA: not demonstrated by quick/local runs.",
        "",
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "architecture_upgrade_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize IWM architecture upgrades against local baselines.")
    parser.add_argument("--out-root", default="outputs/industrial_world_model/architecture_diagnostics")
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    iwm_out = project_root / "outputs" / "industrial_world_model"
    rows: list[dict] = []
    _append_visual_foundation(rows, iwm_out)
    _append_dense_visual(rows, project_root)
    _append_sensor_value(rows, project_root)
    _append_training_stability(rows, iwm_out)
    df = pd.DataFrame(rows)
    out_dir = Path(args.out_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "architecture_upgrade_diagnostics.csv", index=False)
    _write_report(df, out_dir)
    print(f"Architecture diagnostics written to {out_dir}")


if __name__ == "__main__":
    main()
