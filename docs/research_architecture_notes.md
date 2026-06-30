# Research Architecture Notes

Status date: 2026-06-12

This note records architecture implications from four local papers:

- `research_papers/V-JEPA 2.1 Unlocking Dense Features in Video.pdf`
- `research_papers/Intuitive physics understanding emerges from.pdf`
- `research_papers/DINOv3.pdf`
- `research_papers/Self-Supervised Learning from Images with a.pdf`

## Main Conclusion

The current Visual-JEPA MVP is intentionally too small and too global to inherit the strengths of modern
I-JEPA, V-JEPA 2.1, or DINOv3. It predicts one global embedding from a masked image with a small ConvNet.
That is not enough for industrial visual anomaly detection, where dense patch-level features matter.

For visual industrial inspection, the next architecture should be:

1. A patch-token encoder, preferably ViT-like or a frozen DINO-family backbone.
2. A target encoder updated by EMA or a frozen teacher.
3. Position-conditioned mask tokens in the predictor.
4. Losses on patch tokens, not only one global vector.
5. Dense context self-supervision so visible tokens cannot collapse into global aggregators.
6. Intermediate-layer supervision or multi-layer feature concatenation.
7. Patch-similarity regularization inspired by DINOv3 Gram anchoring.
8. Evaluation through PatchCore/PaDiM/kNN-style dense anomaly scoring before claiming JEPA helps.

## I-JEPA

I-JEPA predicts target block representations from an informative context block. Important design points:

- Targets are sampled as several relatively large blocks, not random tiny pixels.
- Context is a large spatially distributed block with target overlaps removed.
- Targets are obtained by masking the output of the target encoder, not by masking the target input.
- The target encoder is updated by EMA from the context encoder.
- The predictor is a narrower ViT-style module, conditioned on target positional mask tokens.
- Average-pooled patch tokens are used for global evaluation rather than relying on a CLS token.

Implication for this repo:

- Replace the current global image-level `VisualJEPA` objective with patch-token context/target prediction.
- Use semantic block masks and output-space target masking.
- Add EMA target encoder. The current MVP uses the same encoder on context and target in one forward pass.

## V-JEPA 2.1

V-JEPA 2.1 fixes a key weakness of V-JEPA-style masked latent prediction: dense feature quality does not
emerge reliably when the loss is applied only to masked tokens.

Key ingredients:

- Dense predictive loss: apply the JEPA loss to masked tokens and visible context tokens.
- Context token loss should be distance-weighted toward nearby masked tokens to avoid trivial copying.
- Deep self-supervision: apply losses to several intermediate encoder layers, not only the final layer.
- Multi-modal tokenizers: separate image and video patch embeddings rather than treating images as static videos.
- High-resolution cooldown improves dense geometry.
- Distillation can compress a large model into smaller deployable models.

Implication for this repo:

- For MVTec/Kolektor/VisA, dense token features are mandatory. The anomaly score should be patch-level.
- Implement `Lpredict + Lctx`, where `Lctx` is distance-weighted by proximity to masked patches.
- Add an encoder API that can return intermediate feature maps.
- Add a multi-layer dense anomaly score: last layer plus selected intermediate layers.

## DINOv3

DINOv3 is important because it focuses directly on robust dense visual features.

Key ingredients:

- Mix global DINO-style loss with local iBOT-style patch loss.
- Use register tokens to reduce high-norm patch outliers.
- Dense feature quality can degrade during long training even while global quality improves.
- Gram anchoring preserves patch-level similarity structure by matching the student patch Gram matrix to a
  teacher Gram matrix.
- High-resolution adaptation with Gram anchoring improves dense tasks and makes features stable across resolutions.
- Distillation from a large teacher into smaller ViT/ConvNeXt students is central to practical deployment.

Implication for this repo:

- Do not train a visual industrial model only by global classification/anomaly loss.
- Add a diagnostic that visualizes patch cosine/PCA maps across resolutions.
- If training from scratch, add a Gram regularizer over patch features.
- If using pretrained features, DINO-family features should be the first strong visual baseline before claiming
  any Visual-JEPA advantage.

## Intuitive Physics / Latent Surprise

The intuitive-physics paper supports latent prediction as a way to learn physical regularities:

- Predict in representation space, not pixel space.
- Compute a temporal surprise score from future latent prediction error.
- Average surprise works well for paired possible/impossible videos.
- Maximum surprise can better separate single videos with local violation events.
- The paper notes a current limitation: V-JEPA is observer-only and lacks action conditioning, which aligns
  with the need for action-conditioned world models in industrial processes.

Implication for this repo:

- The sensor world model direction is conceptually correct, but the adversarial benchmark showed that
  cycle/metadata proxies explain too much of the current result.
- For sensors, use token/time-step latent prediction and local surprise curves, not only one global embedding.
- Report both average and maximum surprise over a tool timeline.
- Keep action-conditioned and no-action variants separate.

## Recommended Next Architecture

### Visual

Implement a `DenseVisualJEPA` path:

- ViT patch encoder or frozen DINO-style backbone.
- EMA target encoder.
- Multi-block target masks.
- Positional mask tokens.
- Patch-token predictor.
- `Lmasked` on target tokens.
- `Lcontext` on visible context tokens with distance weighting.
- Optional multi-layer loss.
- Optional Gram anchoring over patch-token similarities.

Then evaluate:

- Direct JEPA anomaly score.
- PatchCore/kNN on DenseVisualJEPA features.
- PaDiM-style Gaussian patch model.
- DINO-family frozen features + PatchCore/kNN as the reference baseline.

### Sensor

Implement a `DenseSensorJEPA` path:

- Tokenize windows into time/channel patches instead of one global vector.
- EMA target encoder.
- Predict future token embeddings over horizons h.
- Compute average and max latent surprise over cycles.
- Run a strict no-cycle-proxy protocol.

### Decision Rule

Do not prioritize more tuning of the current global-vector Visual-JEPA. It is useful as an MVP, but the
papers point toward dense token features, EMA teachers, intermediate supervision, and strong DINO/PatchCore
baselines as the correct next step.
