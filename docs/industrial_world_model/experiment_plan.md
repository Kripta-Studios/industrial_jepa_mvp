# Experiment Plan

## Phase 1: Visual Foundation

Run visual anomaly detection with:

- pixel/stat baseline;
- DINOv2 dense features from local cache, with explicit fallback if unavailable;
- DINOv3 only as future replacement if weights are available locally;
- PatchCore-lite;
- PaDiM-lite;
- heatmap examples.

## Phase 2: LeJEPA/SIGReg

Train LeJEPA smoke model and check:

- finite loss;
- SIGReg value;
- variance diagnostics;
- collapse flag.

Then compare frozen features against visual baselines.

## Phase 3: Sensor/Process LeJEPA

Use engineered sensor features as the main baseline. LeJEPA must be judged by delta over engineered features, not absolute AUPRC.

## Phase 4: LeWorldModel

Train latent transition predictor:

```text
z_t + action_t + h -> z_{t+h}
```

Evaluate surprise only when temporal structure is valid.

## Phase 5: Hierarchy

Aggregate local scores into image, cycle, lot and line risk tables.

## Phase 6: Product Demo

Generate local HTML demo from real outputs. Missing outputs are marked as missing.
