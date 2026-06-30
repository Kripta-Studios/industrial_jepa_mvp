# Sensor MVP Architecture Assessment

## Decision

For sensors, the product baseline is not a DINO/PatchCore equivalent. The commercial baseline should be:

```text
metadata/cycle + engineered sensor features + calibrated classical model
```

DenseSensorJEPA, predicted future embeddings and world-model surprise should be evaluated as incremental temporal signals. They should not replace engineered features until they improve them consistently.

## Current Evidence

The consolidated CNC results support three conclusions:

- Operational metadata/cycle is strong and unstable deltas make a strong JEPA claim premature.
- No-cycle evaluation shows that physical sensor signals matter when lifecycle/cycle proxies are removed.
- Held-out hardness/material splits are the most promising hard-generalization case for sensor features.

JEPA/future features add small selected deltas over engineered features:

- no-cycle engineered + current z: about `+0.0150` AUPRC;
- no-cycle engineered + predicted future z: about `+0.0087`;
- held-out hardness best combo: about `+0.0346`;
- cutting-condition splits are not consistently improved.

## What Already Exists

The repo already contains a first DenseSensorJEPA path:

- temporal tokenization in `src/sensor_jepa/data/temporal_tokenization.py`;
- span masks in `src/sensor_jepa/data/temporal_masks.py`;
- dense encoder/predictor in `src/sensor_jepa/models/`;
- masked token loss, visible token loss, optional future loss and EMA target encoder;
- surprise scoring with avg/max/top-k and dense pooling diagnostics in `src/sensor_jepa/eval/dense_sensor_surprise.py`;
- IWM surprise helpers for mean/max/top-k/EWMA and residual surprise.

This is a scaffold, not yet a strong sensor product model.

## Main Gaps

1. Future prediction is not yet a proper multi-horizon token world model conditioned on horizon and action/context.
2. DenseSensorJEPA does not yet use SIGReg as the primary anti-collapse comparison.
3. No scheduled span masking or latent re-masking is implemented.
4. No inverse dynamics objective exists, and it should only be used with real actions/setpoints.
5. Surprise curves are evaluated, but not yet fully integrated as lead-time features with EWMA, persistence and first-alert timing.
6. Official MiniROCKET/MultiROCKET/TS2Vec remain required for academic competitiveness.

## Recommended Architecture

Use window tokens:

```text
X_t [C, L] -> temporal/channel tokens -> Z_t [N, D]
```

Train with:

```text
L = L_masked + alpha L_future + beta L_context + lambda L_SIGReg
```

Evaluate four feature groups:

- engineered sensor features;
- dense current token pools;
- future latent predictions;
- surprise features: mean, max, top-k, EWMA, slope, persistence and first alert.

The downstream risk model can remain Logistic/GBT:

```text
risk = model(engineered, z_pool, z_future, surprise, metadata/context)
```

## Protocol Separation

Report three world-model variants separately:

- no-action: `P(Z_t, h) -> Z_t+h`;
- context-conditioned: `P(Z_t, c_t, h) -> Z_t+h`;
- real-action-conditioned: `P(Z_t, a_t, h) -> Z_t+h`.

Only the third should be called action-conditioned. Context variables such as tool, material, hardness or recipe are commercially useful but are not causal action evidence.

## Advancement Criteria

DenseSensorJEPA should move from experimental to product-adjacent only if it:

1. improves engineered features across no-cycle and hard splits;
2. improves Precision@10 or lead time without increasing false alarms materially;
3. shows predicted future features improve current-z features;
4. shows real actions improve no-action variants when real actions exist;
5. transfers or probes well on CWRU/Paderborn/C-MAPSS.

## Product Wording

Use:

> Early-warning risk scoring that combines industrial indicators with self-supervised temporal representations to identify when and how a machine process starts to deviate.

Avoid:

> A JEPA/world-model replacement for engineered features or a proven action-conditioned machine dynamics model.

