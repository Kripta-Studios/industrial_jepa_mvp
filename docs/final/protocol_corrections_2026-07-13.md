# Protocol corrections (2026-07-13)

This repository no longer treats the previous held-out-hardness or visual
thresholded figures as valid post-audit evidence.

## Closed implementation defects

- Numeric hard splits now bin unique raw physical values before assigning rows.
  The split artefact records the exact train/validation/test values and verifies
  raw-value disjointness, not only bin-label disjointness.
- Visual test labels are never used to choose a threshold. MVTec runs reserve a
  deterministic nominal subset of train, fit a declared normal-score quantile
  there, freeze it, and then evaluate test. AUROC/AUPRC are the primary
  threshold-free metrics; F1/balanced accuracy are secondary.
- Requested pretrained backbones fail closed. A missing DINOv2/DINOv3/ResNet
  may use `patch_stats` only after `--allow-fallback`; every row contains
  `requested_backbone`, `actual_backbone`, `pretrained` and `fallback_used`.
- The generated product demo declares `mode=smoke`, `claim_tier=smoke` and
  `comparable_benchmark=false`.

## Evidence status

`results/index.json` intentionally contains no post-audit run. Historical
numbers in presentations/reports are not comparable to the corrected protocol.
A valid rerun must register config/data hashes, the exact physical split values,
the frozen threshold source and the actual executed backbone.

## Reproduction gates

```powershell
python -m pytest -q
python scripts/41_visual_foundation_benchmark.py --config configs/industrial_world_model/visual_foundation_mvtec.yaml --quick
```

The second command fails if the configured backbone is unavailable. For a
dependency-light pipeline smoke (not a DINO result), append
`--allow-fallback` and verify `backbone_info.json` before interpreting output.
