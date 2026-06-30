from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import _bootstrap  # noqa: F401
from common.config import get_device_name, load_config
from common.paths import ensure_dir
from common.reports import markdown_table, write_json, write_markdown_report
from sensor_jepa.data.cnc_milling import prepare_from_config
from sensor_jepa.models.baselines import run_deep_baselines, run_sklearn_baselines
from sensor_jepa.train.finetune import finetune_sensor_jepa
from sensor_jepa.train.pretrain import pretrain_sensor_jepa
from sensor_jepa.train.probe import run_sensor_probe


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sensor_jepa/demo_sensor_quick.yaml")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    bundle = prepare_from_config(cfg, force=False)
    if not Path(cfg["outputs"]["checkpoint"]).exists():
        pretrain_sensor_jepa(cfg)
    fractions = [1.0] if args.quick else [0.05, 0.10, 0.25, 1.0]
    rows = []
    rows.extend(run_sklearn_baselines(bundle.x_train, bundle.y_train, bundle.x_test, bundle.y_test))
    rows.extend(run_deep_baselines(bundle.x_train, bundle.y_train, bundle.x_test, bundle.y_test, device=get_device_name(cfg.get("device", "auto")), epochs=4 if args.quick else 8))
    for frac in fractions:
        rows.append(run_sensor_probe(cfg, "linear", frac))
        rows.append(run_sensor_probe(cfg, "mlp", frac))
        rows.append(finetune_sensor_jepa(cfg, "full", frac))
    out_root = Path(cfg["outputs"]["root"]) / "benchmark"
    ensure_dir(out_root)
    out_csv = out_root / "sensor_benchmark_results.csv"
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    ranking = df.sort_values("macro_F1", ascending=False, na_position="last").copy()
    ranking.insert(0, "rank", range(1, len(ranking) + 1))
    ranking_cols = ["rank", "model_name", "model_family", "macro_F1", "balanced_accuracy", "accuracy"]
    (out_root / "model_ranking.md").write_text(markdown_table(ranking[ranking_cols].to_dict("records")), encoding="utf-8")
    summary = {
        "dataset": "cnc_milling",
        "primary_metric": "macro_F1",
        "best_model": ranking.iloc[0]["model_name"] if len(ranking) else None,
        "best_macro_F1": float(ranking.iloc[0]["macro_F1"]) if len(ranking) else None,
        "sota_claim": False,
        "pending_strong_baselines": ["ROCKET/MiniROCKET", "TS2Vec"],
    }
    write_json(out_root / "sensor_benchmark_summary.json", summary)
    report = out_root / "reports" / "sensor_benchmark_report.md"
    write_markdown_report(report, "Sensor Benchmark Report", {"Results": markdown_table(rows), "Ranking": markdown_table(ranking[ranking_cols].to_dict("records")), "Pending Strong Baselines": "ROCKET/MiniROCKET and TS2Vec are pending because sktime/TS2Vec are not installed."})
    print(out_csv)
    print(report)


if __name__ == "__main__":
    main()
