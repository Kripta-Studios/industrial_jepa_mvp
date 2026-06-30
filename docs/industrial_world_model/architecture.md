# Industrial Predictive Quality World Model Architecture

## Objective

The new line in this repository is a clean product-oriented MVP for industrial predictive quality and anomaly detection. It is separate from the previous `sensor_jepa` and `visual_jepa` modules.

Core idea:

> Learn the expected evolution of normal industrial processes from images, sensors and process parameters, then detect when reality deviates from that expected latent evolution.

## Components

### Visual Foundation

The product baseline starts with dense visual anomaly detection:

- DINOv2 dense features when locally cached.
- DINOv3 only as a future replacement path if local weights become available.
- timm/ResNet-style fallback is reported explicitly.
- PatchCore-lite nearest-neighbor memory.
- PaDiM-lite position-wise Gaussian scoring.
- Pixel/stat baseline for weak-control comparison.
- Image-level anomaly scores and patch heatmaps.

The current code avoids automatic downloads. In the current environment, the active pretrained backbone is DINOv2 `dinov2_vits14` loaded from the local `torch.hub` cache. If that cache is missing, the report marks fallback usage.

### LeJEPA / SIGReg

LeJEPA is implemented as a reusable self-supervised component:

- prediction objective between related views;
- SIGReg isotropic Gaussian regularization;
- no EMA teacher by default;
- no stop-gradient by default;
- primary hyperparameter: `lambda_sigreg`;
- collapse diagnostics: variance, pairwise distance and collapse flag.

### LeWorldModel

The latent world model predicts:

```text
z_hat_{t+h} = predictor(z_t, action_t, h)
```

Where `action_t` can be real setpoints, recipe, material, speed, feed, temperature or process context. If no real actions exist, the report must mark the run as context-based or pseudo-temporal.

### Hierarchical Aggregation

The hierarchy is:

```text
patch -> image/piece -> cycle -> lot -> line
window -> cycle -> machine/tool -> lot/shift
```

Implemented aggregation methods:

- mean;
- max;
- top-k mean;
- EWMA;
- group risk tables;
- top alert ranking.

## Product MVP

The MVP should always deliver value even if LeJEPA or LeWorldModel do not beat baselines:

1. visual anomaly foundation;
2. sensor/process engineered baseline;
3. optional LeJEPA in-domain representation;
4. optional latent surprise;
5. hierarchical alert report;
6. local HTML product demo.
