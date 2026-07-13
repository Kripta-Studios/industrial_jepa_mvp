# Reproducible rerun plan

## Frozen inputs

- Raw hardness group assignments: `artifacts/splits/hardness_raw_split_manifest.json`.
- Experiment config: `configs/sensor_jepa/hard_generalization_cnc.yaml`.
- Data seed: `42`; split seed: `42`; model seeds: `42`, `123`, `999`.
  The membership and `split_hash` remain identical for all model seeds.
- Visual decision threshold: fitted on the nominal training holdout only,
  frozen before test labels are loaded or scored.
- Requested and actual visual backbones must both be recorded. A missing DINO
  model fails closed unless fallback is explicitly enabled; its actual backbone
  is `patch_stats` and the run/model label must be `patch_stats_smoke`, never DINO.
- DINOv3 uses the official Hugging Face model/revision and requires authorized
  gated weights; current blocker is `gated_weights_unauthorized`. DINOv2 must
  use the local `dinov2_vits14` checkpoint with SHA-256
  `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`.

## Gates

1. Start from a clean, committed worktree and record the commit hash.
2. Recreate the split manifest and require identical source/config/split hashes.
3. Require pairwise-disjoint raw hardness, part/workpiece and tool IDs. Record
   every alias and its hash. The frozen connected-component sizes are
   662/69/106 rows, so imbalance must be reported. `MillingToolType` is disclosed
   separately as a coarse, non-identity covariate that cannot form three useful
   partitions.
4. Execute all three frozen seeds once. Do not tune after opening the test set.
5. Report per-seed rows and mean plus standard deviation; no cherry-picking.
6. For visual experiments, persist validation indices, validation scores,
   threshold, threshold source, requested backbone, actual backbone, and
   fallback reason before test evaluation.
   Fit every density model and threshold separately for each MVTec category;
   aggregate only the sealed per-category rows.
7. Run the test suite. Register every output and its SHA-256 digest.
8. Promote to `claim_eligible` only in a separate review after all gates pass.

The completed 2026-07-13 three-seed run is a protocol diagnostic, not the clean
rerun in this plan. It must not be rerun or tuned merely to improve its test
metrics.
