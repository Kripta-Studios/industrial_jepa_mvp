# Current status — 2026-07-13

Status: `recovery_in_progress`; claim-eligible post-audit results: **0**.

- Corrected raw-value grouped splitting is implemented and fingerprinted.
- The local CNC sources were available; a frozen three-seed manifest was
  created and every seed passes raw-value non-overlap.
- A three-seed sensor rerun completed, but its status is `diagnostic_dirty`
  and subsequently failed the fuller identity audit: tools crossed partitions
  and model seeds also changed split membership. It is now invalid evidence.
- The corrected frozen hardness partition uses connected components of raw
  hardness, part and tool IDs: 662/69/106 transition rows. This is leakage-safe
  for those physical identities but severely imbalanced. `MillingToolType` is a
  coarse two-level covariate that connects independent tools; it is disclosed
  in the manifest and is not treated as a physical ID.
- Visual threshold selection is validation-only and test-label mutation tests
  verify that the frozen threshold cannot change.
- The official Hugging Face DINOv3 loader is implemented but its weights are
  gated and this environment is unauthorized (`gated_weights_unauthorized`). It
  fails closed. Explicit substitution has actual backbone `patch_stats`; any
  such run must be labelled `patch_stats_smoke` and is not DINO.
- Public DINOv2 ViT-S/14 is present locally; checkpoint SHA-256 is
  `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`.
  Its real benchmark waits for a clean base commit.
- Multi-category MVTec execution now fits the baseline, PatchCore, PaDiM and
  validation threshold independently per category before aggregation.
- Legacy quantitative claims remain invalidated in README/paper/deck/demo
  surfaces. A clean, hash-identical rerun and independent review remain pending.
