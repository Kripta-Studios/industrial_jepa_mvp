from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import _bootstrap  # noqa: F401
from common.config import get_device_name, load_config
from common.manifest import write_dataset_manifest
from common.paths import ensure_dir
from common.reports import markdown_table, write_markdown_report
from sensor_jepa.data.cnc_milling import prepare_from_config
from sensor_jepa.models.baselines import run_deep_baselines, run_sklearn_baselines
from sensor_jepa.train.evaluate import summarize_sensor_outputs
from sensor_jepa.train.finetune import finetune_sensor_jepa
from sensor_jepa.train.pretrain import pretrain_sensor_jepa
from sensor_jepa.train.probe import run_sensor_probe
from sensor_jepa.train.world_model import evaluate_sensor_world_model, pretrain_sensor_world_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sensor_jepa/demo_sensor_quick.yaml")
    parser.add_argument("--quick", action="store_true", default=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    write_dataset_manifest()
    bundle = prepare_from_config(cfg, force=True)
    ckpt, _ = pretrain_sensor_jepa(cfg, force_data=False)
    rows = [
        run_sensor_probe(cfg, "linear", label_fraction=1.0),
        run_sensor_probe(cfg, "mlp", label_fraction=1.0),
        finetune_sensor_jepa(cfg, mode="full", label_fraction=1.0),
    ]
    rows.extend(run_sklearn_baselines(bundle.x_train, bundle.y_train, bundle.x_test, bundle.y_test))
    rows.extend(run_deep_baselines(bundle.x_train, bundle.y_train, bundle.x_test, bundle.y_test, device=get_device_name(cfg.get("device", "auto")), epochs=4))
    world_ckpt, _, transition_bundle = pretrain_sensor_world_model(cfg)
    world_metrics = evaluate_sensor_world_model(cfg, transition_bundle)
    out_root = Path(cfg["outputs"]["root"])
    ensure_dir(out_root / "benchmark")
    result_csv = out_root / "benchmark" / "sensor_benchmark_results.csv"
    pd.DataFrame(rows).to_csv(result_csv, index=False)
    write_markdown_report(
        out_root / "reports" / "sensor_demo_report.md",
        "Sensor-JEPA Demo Report",
        {
            "Checkpoint": f"`{ckpt}`",
            "Dataset": f"CNC windows train={bundle.x_train.shape}, test={bundle.x_test.shape}",
            "Results": markdown_table(rows),
            "Action World Model": f"Checkpoint: `{world_ckpt}`\n\n{markdown_table([world_metrics])}",
            "Conclusion Rule": "Use macro-F1 and low-label follow-up runs to decide if JEPA adds value. This is not a SOTA claim.",
        },
    )
    summarize_sensor_outputs(cfg)
    print(f"Wrote {result_csv}")


if __name__ == "__main__":
    main()
