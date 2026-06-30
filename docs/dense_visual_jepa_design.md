# Dense Visual JEPA Design

Status date: 2026-06-12

## Why The Current Visual-JEPA Is Obsolete

The current visual MVP is useful as an end-to-end smoke test, but it is not a serious visual anomaly
architecture:

- It uses a small convolutional encoder.
- It predicts one global embedding per image.
- Its heatmaps are derived indirectly from weak feature maps.
- It does not use an EMA target encoder.
- It does not predict patch-token targets conditioned by position.
- It does not supervise visible/context tokens.
- It does not use intermediate-layer supervision.
- It underperforms the simple pixel-stat baseline on MVTec bottle.

This design cannot make a credible claim against DINO-family frozen features, PatchCore, PaDiM, or modern
V-JEPA/DINO-style dense feature methods.

## What Changes With DenseVisualJEPA

DenseVisualJEPA shifts the visual path from global image embeddings to dense patch-token representations.
The goal is not to reconstruct pixels, but to learn patch-level latent features that can support industrial
anomaly scoring.

Implemented in this phase:

- ViT-style patch encoder returning `[B, N, D]` dense tokens.
- Optional intermediate hidden states for deep supervision.
- Block-based target masks instead of purely random patches.
- EMA target encoder mode.
- Position-conditioned dense predictor.
- Masked target-token latent loss.
- Optional visible/context-token latent loss.
- Optional deep-supervision loss.
- Patch-level feature extraction.
- kNN/PatchCore-lite memory scoring.
- PaDiM-lite scoring.
- Optional DINO-family wrapper with graceful fallback.
- Configs for MVTec quick, MVTec all, MVTec+VisA+Kolektor, DINOv2 PatchCore, and dense visual benchmark.
- Tests for patching, block masks, DenseVisualJEPA shapes, EMA gradients, dense losses, feature memory, and PaDiM-lite.
- Commercial presentation updates that explain why dense visual features are the correct next visual direction without claiming that they have already won benchmarks.

## What Remains Pending

- Official DINOv3 weights are not assumed to be available locally.
- DINO-target JEPA distillation is optional and not part of the first MVP.
- High-resolution cooldown/adaptation is pending.
- Full multi-seed visual benchmarking is pending unless explicitly run.
- Published SOTA comparison is pending.
- PRO score is optional and not part of the first minimal report.
- DINO-target JEPA distillation is pending until the dense benchmark shows a useful student/teacher tradeoff.
- DenseVisualJEPA results are pending until `scripts/22_pretrain_dense_visual_jepa.py` and `scripts/26_benchmark_dense_visual_jepa.py` are run on available local datasets.

## Allowed Claims

Allowed after tests pass:

- DenseVisualJEPA is implemented as a patch-token visual JEPA MVP.
- DenseVisualJEPA can be compared against pixel-stat, ResNet kNN/PatchCore-lite, and PaDiM-lite.
- If benchmarked, reports may say whether dense features improve over the old global Visual-JEPA on the
  executed categories.

Not allowed:

- No SOTA claim.
- No DINOv3 claim unless official weights are actually available and used.
- No claim that latent prediction error is the best anomaly scorer unless it beats kNN/PaDiM baselines.
- No claim that DenseVisualJEPA is production-ready without multi-category validation and strong baselines.

## First MVP Decision Rule

If DenseVisualJEPA latent-error loses but DenseVisualJEPA+kNN improves, the encoder is useful and the
direct prediction error is not a good scorer.

If DenseVisualJEPA+kNN loses badly to ResNet/DINO PatchCore-lite, the product path should use the stronger
frozen backbone while DenseVisualJEPA remains a research/adaptation path.

If DenseVisualJEPA+kNN approaches or beats strong frozen backbones on several categories, it becomes a
technical candidate worth expanding with DINO-target distillation and high-resolution adaptation.

## Smoke Benchmark Executed

Command:

```powershell
python scripts/26_benchmark_dense_visual_jepa.py --config configs/visual_jepa/dense_mvtec_bottle_quick.yaml
```

Scope:

- Dataset: MVTec AD bottle quick.
- DenseVisualJEPA pretraining: 2 epochs, 64 normal train images, 16 validation images.
- Test set: full bottle test split after removing the previous `max_test_images` cap that selected only anomaly rows.
- This is a smoke benchmark, not a final visual benchmark.

Observed image-level AUPRC/AUROC:

| Model | Image AUROC | Image AUPRC | Pixel AUROC | Pixel AUPRC |
| --- | ---: | ---: | ---: | ---: |
| pixel_stat_baseline | 0.7778 | 0.9205 | 0.8600 | 0.3827 |
| dense_visual_jepa_latent_error | 0.5270 | 0.7830 | 0.4512 | 0.0497 |
| dense_visual_jepa_knn | 0.7635 | 0.9219 | 0.8975 | 0.3285 |
| dense_visual_jepa_padim | 0.8159 | 0.9408 | 0.8866 | 0.3703 |
| resnet18_knn | 0.9881 | 0.9963 | 0.9581 | 0.4682 |
| resnet18_padim | 0.9889 | 0.9965 | 0.9466 | 0.4460 |

Interpretation:

- Dense latent prediction error is weak as a direct anomaly scorer.
- DenseVisualJEPA dense features become useful when scored with kNN/PaDiM-lite.
- ResNet18 frozen features with kNN/PaDiM-lite dominate this quick run.
- Product-facing visual anomaly detection should keep ResNet/DINO/PatchCore-style baselines as the practical reference, while DenseVisualJEPA remains a research/adaptation path until it closes the gap on multiple categories.
