# Experiments

## Sensor-JEPA Demo

```powershell
python scripts/15_run_sensor_demo.py --config configs/sensor_jepa/demo_sensor_quick.yaml
```

Runs:

1. CNC feature preparation into cycle windows split by tool.
2. Sensor-JEPA self-supervised pretraining.
3. Frozen linear and MLP probes.
4. Full fine-tuning.
5. Baselines: Logistic Regression, Random Forest, HistGradientBoosting, MLP, 1D CNN, GRU in quick mode.
6. Action-conditioned latent world-model pretraining:
   `z(window_t), action_t -> z(window_t+h)`.
7. Failure-soon evaluation using latent prediction error as a surprise score.
8. Markdown/CSV reports.

Primary metric: macro-F1 for CNC wear-stage classification.

World-model metric: AUROC/AUPRC for `CycleToFailure <= failure_horizon_cycles`.

Standalone world-model run:

```powershell
python scripts/20_train_sensor_world_model.py --config configs/sensor_jepa/demo_sensor_quick.yaml
```

## Visual-JEPA Demo

```powershell
python scripts/16_run_visual_demo.py --config configs/visual_jepa/demo_visual_quick.yaml
```

Runs:

1. MVTec AD manifest for one category.
2. Visual-JEPA pretraining on normal train images.
3. Embedding anomaly evaluation on test images.
4. Pixel-stat anomaly baseline.
5. Heatmap overlays and Markdown/CSV reports.

Primary metrics:

- Image AUROC and AUPRC.
- Pixel AUROC and AUPRC when ground-truth masks exist.

## Dense Visual-JEPA Patch-Level Phase

The old visual MVP is kept for regression/smoke testing, but the visual research path now moves to dense
patch-token features. These commands do not imply a SOTA claim; they create a reproducible route to compare
DenseVisualJEPA against pixel-stat, ResNet kNN/PatchCore-lite, PaDiM-lite and DINO-family frozen features
when DINO weights are locally available.

Pretrain DenseVisualJEPA quick:

```powershell
python scripts/22_pretrain_dense_visual_jepa.py --config configs/visual_jepa/dense_mvtec_bottle_quick.yaml
```

Pretrain DenseVisualJEPA multi-category:

```powershell
python scripts/22_pretrain_dense_visual_jepa.py --config configs/visual_jepa/dense_mvtec_all.yaml
```

Extract dense features:

```powershell
python scripts/23_extract_dense_visual_features.py --config configs/visual_jepa/dense_mvtec_all.yaml --backbone dense_visual_jepa
```

Evaluate kNN/PatchCore-lite memory anomaly scoring:

```powershell
python scripts/24_eval_visual_memory_anomaly.py --config configs/visual_jepa/dense_mvtec_all.yaml --backbone dense_visual_jepa
```

Evaluate DINO/PatchCore-lite if DINOv2 is available through local cache or network:

```powershell
python scripts/24_eval_visual_memory_anomaly.py --config configs/visual_jepa/dinov2_patchcore.yaml --backbone dinov2
```

Evaluate PaDiM-lite:

```powershell
python scripts/25_eval_visual_padim.py --config configs/visual_jepa/dense_mvtec_all.yaml --backbone dense_visual_jepa
```

Run full dense visual benchmark:

```powershell
python scripts/26_benchmark_dense_visual_jepa.py --config configs/visual_jepa/dense_visual_benchmark.yaml
```

## Benchmark Scripts

```powershell
python scripts/18_benchmark_sensor_models.py --config configs/sensor_jepa/demo_sensor_quick.yaml --quick
python scripts/19_benchmark_visual_models.py --config configs/visual_jepa/demo_visual_quick.yaml --quick
python scripts/21_sota_sensor_world_benchmark.py --config configs/sensor_jepa/demo_sensor_quick.yaml --quick
```

The benchmark layer is designed to answer whether JEPA helps over simple baselines.
Strong domain baselines such as PatchCore/PaDiM and ROCKET/MiniROCKET are marked
pending when their packages are not available.

## Sensor World Model SOTA-Candidate Benchmark

Quick smoke run:

```powershell
python scripts/21_sota_sensor_world_benchmark.py --config configs/sensor_jepa/demo_sensor_quick.yaml --quick
```

Focused 3-seed validation for the current strongest setup:

```powershell
python scripts/21_sota_sensor_world_benchmark.py --config configs/sensor_jepa/demo_sensor_quick.yaml --seeds 42,123,999 --horizons 3 --targets 10 --out-root outputs/sensor_jepa/sota_benchmark_h3_k10_3seed
```

Outputs:

- `sota_benchmark_results.csv`
- `results_by_seed.csv`
- `results_by_tool.csv`
- `results_by_horizon.csv`
- `ablation_results.csv`
- `calibration_metrics.csv`
- `leakage_report.json`
- `sota_candidate_report.md`
- `plots/`

Important limitation: exact MiniROCKET/MultiROCKET are not available unless
`sktime`/`aeon` is installed. The current benchmark records `minirocket_lite`
and `multirocket_lite` as fallbacks, not as published-equivalent baselines.

Adversarial 3-seed grid:

```powershell
python scripts/21_sota_sensor_world_benchmark.py `
  --config configs/sensor_jepa/demo_sensor_quick.yaml `
  --seeds 42,123,999 `
  --horizons 1,3,5,10,20 `
  --targets 5,10,20 `
  --out-root outputs/sensor_jepa/sota_benchmark_adversarial_3seed
```

Additional outputs:

- `feature_audit.csv`
- `forbidden_columns_report.json`
- `action_only_results.csv`
- `cycle_only_results.csv`
- `metadata_only_results.csv`
- `sensor_only_results.csv`
- `sensor_plus_actions_results.csv`
- `horizon_target_grid.csv`
- `horizon_target_heatmap.png`
- `worst_tool_report.md`
- `calibration_protocol.json`
- `sota_validation_report.md`

Current adversarial verdict:

- Leakage report passes.
- Feature audit passes.
- Best single row is not the action-conditioned predicted-future model; it is
  `world_model_current_z_no_actions_scratch` at about 0.852 AUPRC.
- A metadata/cycle baseline is very strong:
  `metadata_only_no_sensor_logistic_regression` reaches about 0.838 AUPRC in
  its best h/K row and is the strongest key model on grid-average AUPRC.
- The action-conditioned predicted-future world model remains useful, but the
  current evidence is not enough for a SOTA claim.

## Sensor Incremental Value Phase

Incremental value benchmark:

```powershell
python scripts/22_incremental_sensor_value_benchmark.py --config configs/sensor_jepa/incremental_value_cnc.yaml
```

Hard generalization:

```powershell
python scripts/23_sensor_hard_generalization.py --config configs/sensor_jepa/hard_generalization_cnc.yaml
```

JEPA vs engineered sensor feature value:

```powershell
python scripts/35_jepa_vs_engineered_value.py --config configs/sensor_jepa/incremental_value_cnc.yaml --seeds 42,123,999 --horizons 1,3,5 --targets 5,10,20 --out-root outputs/sensor_jepa/jepa_vs_engineered_value
```

Check official baselines:

```powershell
python scripts/24_install_check_sensor_baselines.py
```

Pretrain DenseSensorJEPA:

```powershell
python scripts/25_pretrain_dense_sensor_jepa.py --config configs/sensor_jepa/dense_sensor_cnc.yaml
```

Evaluate DenseSensor surprise:

```powershell
python scripts/26_eval_dense_sensor_surprise.py --config configs/sensor_jepa/dense_sensor_cnc.yaml
```

Optional token world model:

```powershell
python scripts/27_train_token_sensor_world_model.py --config configs/sensor_jepa/dense_sensor_cnc.yaml
```

CWRU benchmark placeholder:

```powershell
python scripts/28_benchmark_cwru_sensor_representations.py --config configs/sensor_jepa/cwru_bearing.yaml
```

Paderborn benchmark placeholder:

```powershell
python scripts/29_benchmark_paderborn_sensor_representations.py --config configs/sensor_jepa/paderborn_bearing.yaml
```

Tests:

```powershell
python -m compileall -q src scripts
python -m pytest -q
```

Interpretation rule: do not use absolute AUPRC alone as evidence that Sensor-JEPA
helps. The primary evidence is delta against metadata-only within the same
protocol, seed, forecast horizon and failure target.

For the JEPA-vs-engineered phase, the primary evidence is delta against
`sensor_engineered_only` and `metadata_plus_sensor_engineered` within the same
protocol, seed, forecast horizon and failure target.
