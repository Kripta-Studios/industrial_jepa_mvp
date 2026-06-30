# Industrial JEPA MVP

This repository implements two reproducible MVP demos:

- Sensor-JEPA for predictive maintenance on CNC tool-wear cycle features.
- Visual-JEPA for industrial visual anomaly detection on MVTec AD.

The implementation is intentionally compact. It prioritizes local datasets, frozen
probes, simple baselines, metrics, reports, and repeatable commands over large
foundation-model scale training.

## Quick Commands

```powershell
python scripts/00_create_dataset_manifest.py
python scripts/15_run_sensor_demo.py --config configs/sensor_jepa/demo_sensor_quick.yaml
python scripts/20_train_sensor_world_model.py --config configs/sensor_jepa/demo_sensor_quick.yaml
python scripts/16_run_visual_demo.py --config configs/visual_jepa/demo_visual_quick.yaml
python scripts/17_run_all_demos.py
```

## Scope

Implemented end-to-end:

- CNC milling feature windows from `data/raw/sensor/cnc_milling`.
- Sensor-JEPA pretraining, frozen probes, fine-tuning, and baseline comparison.
- Action-conditioned latent world-model training for CNC transitions
  (`window_t + process_action_t -> window_t+h`) with failure-soon scoring.
- MVTec AD image loading from `data/raw/visual/mvtec_ad`.
- Visual-JEPA pretraining, embedding anomaly scoring, heatmaps, and a pixel-stat baseline.
- Dataset manifest generation for all local Fase 1 datasets.

Partial support:

- CWRU bearing, Paderborn bearing, VisA, and KolektorSDD are scanned and manifested.
- Loader scaffolding exists for these datasets, but the default MVP trains on CNC and MVTec.

Pending:

- Multi-sensor CNC, NASA IMS, C-MAPSS, AI4I, MVTec 3D-AD, NEU, DAGM, Severstal,
  and wood defects unless real files are present.

## No SOTA Claim

Do not claim "SOTA" from these outputs. Use language such as:

- "MVP competitive against internal baselines."
- "Promising in low-label settings."
- "Not validated as SOTA."

A SOTA claim would require standard splits, strong baselines such as PatchCore,
PaDiM, ROCKET/MiniROCKET, multiple seeds, validation-only threshold selection,
published-number comparison, and reproducible timing.
