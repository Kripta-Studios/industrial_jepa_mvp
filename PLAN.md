# Industrial JEPA MVP Plan

Status date: 2026-06-12

## What happened to PLAN.md

There was no `PLAN.md` file in the workspace when the implementation started.
The operative plan was in `PROMPT.md`, so the work was executed from that file.
This `PLAN.md` now records the implemented scope, current evidence, and next
steps.

## SOTA Notes From Research Papers

The local papers point to these design requirements:

- JEPA should predict latent embeddings, not reconstruct raw pixels or signals.
- Collapse prevention is required; this MVP uses variance/SIGReg-style regularization.
- Frozen probes are necessary to test whether the representation is reusable.
- LeWorldModel adds the missing forecasting piece: latent state plus action/context
  predicts future latent state.
- I-JEPA/V-JEPA 2.1/DINOv3 indicate that visual SOTA requires dense patch-token
  features, EMA/frozen target encoders, position-conditioned mask tokens,
  context-token self-supervision, intermediate-layer supervision, and/or patch
  similarity regularization.
- The intuitive-physics results support latent future surprise as a useful signal,
  but the industrial protocol must separate real dynamics from cycle/metadata
  proxies.
- No SOTA claim is valid without strong baselines, standard splits, multiple seeds,
  and published-number comparison.

Detailed architecture notes are recorded in
`docs/research_architecture_notes.md`.

## Implemented

### Dataset and manifests

- `data/manifests/datasets.yaml`
- `data/manifests/datasets.csv`
- `data/manifests/cnc_windows.csv`
- `data/manifests/mvtec_bottle.csv`

Dataset states:

- `cnc_milling`: implemented.
- `mvtec_ad`: implemented.
- `cwru_bearing`: partial manifest/inspection.
- `paderborn_bearing`: partial manifest/inspection.
- `visa`: partial manifest loader.
- `kolektor_sdd`: partial manifest loader.
- Missing/pending: multi-sensor CNC, NASA IMS, C-MAPSS, AI4I, MVTec 3D-AD,
  NEU, DAGM, Severstal, wood defects.

### Sensor-JEPA

- CNC cycle-window loader with split by tool.
- Conv1D JEPA encoder/predictor.
- Latent prediction loss plus variance regularization.
- Frozen linear probe.
- Frozen MLP probe.
- Full fine-tune.
- Baselines: Logistic Regression, Random Forest, HistGradientBoosting, MLP,
  supervised 1D CNN, supervised GRU.
- Reports and benchmark CSVs.

### Sensor action-conditioned world model

Implemented after review of the LeWorldModel requirement:

- Current state: CNC sensor-feature window.
- Action/context: `MillingToolType`, `ADOC`, `RDOC`, `HardnessMean`,
  `ToolHolderLength`, `ToolRotation`, `FeedRate`, `ToolDiameter`.
- Objective: predict future latent embedding `z(window_t+h)` from
  `z(window_t)` and `action_t`.
- Failure-soon target: `CycleToFailure <= 10`.
- Readout: logistic probe on predicted future embeddings.

### Visual-JEPA

- MVTec AD loader for category-specific train/test/mask manifests.
- Small Conv visual encoder.
- Masked latent prediction.
- Patch-level anomaly heatmaps.
- Pixel-stat anomaly baseline.
- Reports and overlays.

## Current Metrics

### Sensor wear classification, CNC

Latest quick benchmark:

- Best Sensor-JEPA variant: `sensor_jepa_full_finetune`, macro-F1 about 0.53.
- Frozen probes are weak, macro-F1 about 0.33 to 0.34.
- Classic/deep baselines remain competitive.

Interpretation:

- JEPA full fine-tune is useful as a trained model.
- Reusable frozen representation is not yet proven.

### Sensor failure-soon forecasting, CNC world model

Latest world-model evaluation:

- Raw latent prediction error AUROC: about 0.255.
- Predicted-future-embedding probe AUROC: about 0.828.
- Predicted-future-embedding probe AUPRC: about 0.470.
- Test failure rate: about 0.084.

Interpretation:

- The raw surprise/error score does not generalize as a failure signal on this split.
- The predicted future latent state contains useful information for failure-soon
  risk when read through a lightweight supervised probe.
- This is the LeWorldModel-style component of the MVP.

### SOTA-candidate sensor benchmark

Implemented after MVP:

- Multi-seed CLI: `scripts/21_sota_sensor_world_benchmark.py`.
- Targets: configurable `CycleToFailure <= K`.
- Horizons: configurable `h`.
- Anti-leakage report by disjoint tool splits.
- Metrics: AUROC, AUPRC, Precision@5/10%, Recall@5/10%, F1,
  balanced accuracy, Brier score, ECE, false alarms per tool, lead time.
- Baselines: Logistic Regression, Random Forest, HistGradientBoosting,
  XGBoost, LightGBM, CNN1D, GRU, TCN, MiniROCKET-lite, MultiROCKET-lite,
  TS2Vec proxy, world-model ablations.
- Outputs under `outputs/sensor_jepa/sota_benchmark*`.

Focused 3-seed run for `h=3`, `K=10`:

- Best model: `world_model_pred_future_actions_calibrated`.
- Mean AUPRC: about 0.770.
- Mean AUROC: about 0.922.
- Precision@10% alerts: about 0.770.
- Recall@10% alerts: about 0.503.
- Leakage check: pass, no tool overlap.

Important limitation:

- MiniROCKET/MultiROCKET are currently fallback lite implementations because
  `sktime`/`aeon` are not installed.
- TS2Vec is a proxy temporal contrastive encoder, not official TS2Vec.
- Therefore this is a SOTA-candidate benchmark scaffold, not a SOTA claim.

Adversarial 3-seed grid run:

- Command:
  `python scripts/21_sota_sensor_world_benchmark.py --config configs/sensor_jepa/demo_sensor_quick.yaml --seeds 42,123,999 --horizons 1,3,5,10,20 --targets 5,10,20 --out-root outputs/sensor_jepa/sota_benchmark_adversarial_3seed`
- Rows: 1350 benchmark rows, 30 model names.
- Seeds: 42, 123, 999.
- Horizons: 1, 3, 5, 10, 20.
- Targets: 5, 10, 20.
- Leakage check: pass, disjoint tools across train/val/test.
- Feature audit: pass, no forbidden target/RUL/life-stage columns used by
  encoder or action vector.
- Best single h/K/model row: `world_model_current_z_no_actions_scratch`,
  AUPRC about 0.852 at h=10/K=20.
- Best predicted-future action world model row:
  `world_model_pred_future_actions_scratch`, AUPRC about 0.782 at h=10/K=20.
- Best calibrated predicted-future action world model row:
  `world_model_pred_future_actions_calibrated`, AUPRC about 0.721 at h=3/K=20.
- Strong adversarial baseline:
  `metadata_only_no_sensor_logistic_regression`, AUPRC about 0.838 at h=10/K=20.
- Grid average across all h/K/seeds:
  `metadata_only_no_sensor_logistic_regression` is strongest among key models
  with mean AUPRC about 0.671.

Adversarial interpretation:

- The original h=3/K=10 signal remains useful, but it is not sufficient for a
  SOTA claim.
- The grid shows that cycle/metadata-only baselines explain a large part of the
  apparent performance.
- Predicted future embeddings with actions do not consistently beat current
  embeddings or no-action world models.
- This is now a strong MVP and diagnostic benchmark, not a validated SOTA
  candidate yet.

### Visual anomaly detection, MVTec bottle

Latest quick benchmark:

- Visual-JEPA image AUROC: about 0.67.
- Pixel-stat baseline image AUROC: about 0.80.
- Visual-JEPA pixel AUROC: about 0.70.
- Pixel-stat baseline pixel AUROC: about 0.86.

Interpretation:

- Visual-JEPA is implemented end-to-end but does not beat the simple baseline yet.
- PatchCore/PaDiM are still pending.

## Commands

Prepare manifest:

```powershell
python scripts/00_create_dataset_manifest.py
```

Sensor demo:

```powershell
python scripts/15_run_sensor_demo.py --config configs/sensor_jepa/demo_sensor_quick.yaml
```

Sensor action world model:

```powershell
python scripts/20_train_sensor_world_model.py --config configs/sensor_jepa/demo_sensor_quick.yaml
```

Visual demo:

```powershell
python scripts/16_run_visual_demo.py --config configs/visual_jepa/demo_visual_quick.yaml
```

All demos:

```powershell
python scripts/17_run_all_demos.py
```

Tests:

```powershell
python -m pytest -q
```

## MVP Verdict

- Sensor-JEPA MVP: Partial/Yes for demo, because it trains end-to-end and has
  benchmarks, but frozen representation value is not proven.
- Sensor action world model: Yes for MVP evidence, because predicted future
  embeddings give useful failure-soon signal.
- Visual-JEPA MVP: Partial, because it trains end-to-end and produces heatmaps,
  but does not beat the baseline yet.
- SOTA claim: No.

## Next Steps

1. Run the new sensor incremental value benchmark:
   `scripts/22_incremental_sensor_value_benchmark.py`.
2. Use delta over metadata-only as the primary CNC Sensor-JEPA evidence.
3. Build a stricter no-cycle-index protocol and rerun the same grid.
4. Separate commercial protocol with known cycle count from paper protocol
   without cycle-position proxies.
5. Use `scripts/23_sensor_hard_generalization.py` to test held-out tool and
   operating-condition generalization.
6. Use `scripts/24_install_check_sensor_baselines.py` before any SOTA claim;
   fallback ROCKET/TS2Vec rows are not claim-eligible.
7. Train/evaluate DenseSensorJEPA token-level surprise only as incremental value
   over metadata/cycle, not by absolute AUPRC alone.
8. Replace the global-vector Visual-JEPA MVP with a dense patch-token
   Visual-JEPA path inspired by I-JEPA/V-JEPA 2.1.
9. Add DINO-family frozen features plus PatchCore/kNN as the primary visual
   reference baseline.
10. Replace MiniROCKET-lite/MultiROCKET-lite with exact `sktime`/`aeon` baselines.
11. Replace TS2Vec proxy with official TS2Vec or a validated implementation.
12. Add a model-selection split/protocol so h/K selection is not chosen from test
   summaries.
13. Add PatchCore or PaDiM for MVTec.
14. Extend world-model evaluation to Paderborn and CWRU where forecasting labels
   are available or define transfer-classification tasks.
15. Improve visual scoring or switch MVTec category if bottle is too easy for
  pixel-stat baselines.
