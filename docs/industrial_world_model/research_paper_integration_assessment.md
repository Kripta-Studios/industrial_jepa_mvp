# Research Paper Integration Assessment

This note summarizes how the requested JEPA/world-model papers should influence the `industrial_jepa_mvp` roadmap. It is based on the local PDFs in `research_papers/` and the current project results.

## Executive Decision

The product should remain positioned as an industrial validation MVP with strong baselines:

- visual anomaly detection with DINOv2/ResNet dense features, PatchCore-lite, PaDiM-lite, heatmaps, and ranking;
- sensor/process risk scoring with metadata, engineered features, and optional latent features;
- LeJEPA, DenseVisualJEPA, DenseSensorJEPA, and LeWorldModel as experimental modules that must prove incremental value.

The papers support improving the architecture, but they do not justify selling a JEPA-first claim.

## Paper-by-Paper Assessment

| Paper | Useful idea | Where it fits | Priority | Product status |
|---|---|---:|---:|---|
| V-JEPA 2.1 | Dense predictive loss, visible/context-token supervision, deep self-supervision, dense features | DenseVisualJEPA and visual foundation adaptation | High | Research/product-adjacent |
| Intuitive physics from video prediction | Latent surprise over time from predicted vs observed representations | LeWorldModel, sensor/process timelines, predictive quality | High | Product-adjacent if validated |
| A lightweight library for energy-based JEPA | Modular encoder/predictor/regularizer/planner design; action-conditioned world models | Industrial World Model code structure and experiment registry | Medium-high | Engineering guidance |
| FF-JEPA | Hierarchical subgoals for long-horizon latent planning | Future cycle/lote/line hierarchy | Medium-low now | Future research |
| Var-JEPA | Latent uncertainty and selective prediction | Calibration, abstention, risk-coverage, human review routing | Medium-low now | Future research |
| JEPA-DNA | Scheduled span masking, predictor warmup, hybrid token-level losses | DenseSensorJEPA token/time masking and possibly DenseVisualJEPA masking schedules | Medium | Inspiration only |

## Recommended Integrations

### 1. Visual Product Core

Use the current strongest route as the product-facing visual MVP:

```text
DINOv2 / ResNet dense features
+ PatchCore-lite / PaDiM-lite
+ image anomaly score
+ pixel/patch heatmap
+ alert ranking
```

DINOv3 should remain an optional future backend until real local weights are available. Reports must state the actual backbone used.

### 2. DenseVisualJEPA Upgrade

V-JEPA 2.1 and I-JEPA-style lessons should drive the next DenseVisualJEPA iteration:

- patch tokens, not global embeddings;
- block target masks;
- positional predictor;
- target encoder EMA or frozen teacher as a clearly labeled variant;
- visible/context token loss;
- deep supervision from intermediate layers;
- multi-layer features for kNN/PaDiM;
- optional DINOv2 teacher distillation after the baseline is stable.

The primary anomaly scorer should be kNN/PaDiM on dense features. Direct latent prediction error is not yet reliable enough to be product-facing.

### 3. LeJEPA / SIGReg Infrastructure

SIGReg-style regularization is useful as a training stability component:

- collapse diagnostics;
- variance/covariance/rank proxies;
- embedding distribution checks;
- one main regularization weight.

This should be used internally for LeJEPA visual, sensor LeJEPA, and LeWorldModel experiments. It should not be sold as a product claim unless downstream metrics improve.

### 4. Sensor and Process Surprise

The intuitive-physics paper supports using prediction error as a temporal surprise signal, but only as measured deviation:

```text
observed latent future - predicted latent future -> surprise curve
```

Recommended outputs:

- mean/max/top-k surprise;
- EWMA surprise;
- surprise normalized by machine/tool/regime;
- residual surprise after metadata/engineered baselines;
- lead-time and false-alarm metrics.

Do not claim causality or autonomous control.

### 5. Hierarchical Industrial World Model

FF-JEPA should not be implemented as full planning in the first product MVP. The transferable idea is hierarchy:

```text
patch -> image/piece -> cycle -> lot -> line
```

Use it first for aggregation, alert ranking, and expected trajectory summaries. Add latent subgoal planning only when real long-horizon process sequences and actions/setpoints exist.

### 6. Uncertainty and Human Review

Var-JEPA is valuable for uncertainty-aware predictive quality, but full variational JEPA is not urgent. Start with simpler production-friendly uncertainty:

- calibrated probabilities;
- ECE/Brier;
- model ensembles;
- kNN distance confidence;
- risk-coverage curves;
- "send to human review" thresholds.

If these are useful, Var-JEPA can become a later research module.

### 7. DenseSensorJEPA

JEPA-DNA does not transfer by domain, but its token-level training pattern is useful:

- span masking;
- scheduled masking;
- predictor warmup;
- future token prediction;
- hybrid local/global losses.

Use this only after the sensor baseline remains anchored on engineered features and no-cycle/hard-generalization deltas.

## What Not To Do Now

- Do not sell JEPA as better than engineered features without delta evidence.
- Do not replace PatchCore/PaDiM with DenseVisualJEPA until DenseVisualJEPA wins or clearly complements them.
- Do not implement FF-JEPA planning before having real long-horizon action/process data.
- Do not implement full Var-JEPA before simpler uncertainty/calibration is evaluated.
- Do not use JEPA-DNA as a direct industrial claim.
- Do not claim DINOv3 unless DINOv3 weights are actually loaded.

## Product Roadmap

### Immediate MVP

1. DINOv2/ResNet + PatchCore-lite/PaDiM-lite.
2. Heatmaps and top anomaly ranking.
3. Sensor/process baselines with engineered features.
4. Demo HTML/PDF and pilot workflow.

### Technical Differentiation

1. DenseVisualJEPA with V-JEPA 2.1 style losses.
2. LeJEPA/SIGReg in-domain pretraining.
3. LeWorldModel surprise timelines on sequential process data.
4. Hierarchical alert aggregation.

### Later Research

1. DINOv3 backend when weights are available.
2. Var-JEPA uncertainty-aware predictive quality.
3. FF-JEPA long-horizon industrial subgoal modeling.
4. DenseSensorJEPA token-time models with scheduled span masking.

## Final Positioning

Use this wording:

> Industrial Predictive Quality World Model is a pilot-ready research MVP that starts from strong visual and sensor baselines, then tests whether in-domain JEPA and latent world-model modules add incremental value for each client process.

Avoid this wording:

> JEPA is a superior industrial foundation model or a production-ready autonomous world model.

