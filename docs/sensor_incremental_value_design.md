# Sensor Incremental Value Design

Status date: 2026-06-12

## Why The Action-Conditioned World-Model SOTA Claim Does Not Hold

The adversarial validation in
`outputs/sensor_jepa/sota_benchmark_adversarial_3seed/` is now the controlling
evidence for CNC failure-soon forecasting.

Key facts:

- Leakage report passes.
- Feature audit passes.
- Encoder/action inputs do not include direct forbidden targets such as
  `CycleToFailure`, RUL, labels or direct derivatives.
- The grid contains 1350 rows across seeds, horizons and targets.
- Best global row is `world_model_current_z_no_actions_scratch` with AUPRC
  about 0.8524.
- Best predicted-future action world model reaches AUPRC about 0.7824.
- Best calibrated predicted-future action world model reaches AUPRC about
  0.7206.
- A strong metadata/cycle baseline,
  `metadata_only_no_sensor_logistic_regression`, reaches AUPRC about 0.8376 in
  its best h/K row and is strongest among key models by grid-average AUPRC.

This means the original strong claim, "action-conditioned latent world model is
SOTA", does not survive. The world model remains useful as an MVP component and
research path, but the evidence does not show stable value over current
embeddings or metadata/cycle baselines.

## Why Metadata/Cycle Is A Strong Baseline

CNC wear/failure-soon labels are tightly coupled to operational state:

- current/source cycle index;
- tool identity and tool type;
- cutting depth and holder length;
- hardness regime;
- process settings.

Those variables are often available in a factory and are commercially useful.
They can also dominate academic evaluation if not separated from raw sensor
signals. Therefore metadata/cycle must be treated as a first-class baseline, not
as nuisance context.

## Main Metric: Delta Over Metadata-Only

The new benchmark reports absolute metrics, but the central evidence is:

- `delta_AUPRC_vs_metadata_only`
- `delta_AUROC_vs_metadata_only`
- `delta_Precision@10_vs_metadata_only`
- `delta_Recall@10_vs_metadata_only`
- `delta_false_alarms_vs_metadata_only`
- `delta_lead_time_vs_metadata_only`

Deltas are computed only inside the same protocol, seed, forecast horizon `h`
and failure target `K`. This avoids comparing a best global row from one setting
against a baseline from another.

## Commercial Protocol vs Academic Protocol

Commercial protocol:

- may use metadata available at operation time;
- may use known cycle count if the factory has it;
- answers whether a deployed risk-scoring system is useful.

Academic protocol:

- must isolate raw sensor value from cycle/life proxies;
- should include no-cycle and hard-generalization splits;
- should compare against official strong baselines such as MiniROCKET,
  MultiROCKET and TS2Vec when available;
- cannot claim reusable representation unless frozen probes or hard transfer
  support it.

## Implemented Now

- `incremental_value_benchmark`:
  - metadata-only, cycle-only, actions-only;
  - sensor raw only;
  - current-z and predicted-future-z features from the existing world-model
    route;
  - metadata plus sensor/JEPA/world-model combinations;
  - same validation-threshold protocol as the adversarial benchmark;
  - deltas against metadata-only in matched groups.
- GBT vs JEPA feature value analysis:
  - GBT metadata-only;
  - GBT engineered sensor features;
  - GBT metadata plus engineered sensor features;
  - GBT current-z/predicted-future-z combinations;
  - native feature importance when the estimator exposes it.
- Hard generalization scaffold:
  - held-out tool id/type;
  - hardness/feed/rotation/holder bins when columns have enough groups;
  - cutting-condition groups when ADOC/RDOC exist;
  - missing or degenerate splits are marked `pending`.
- Official baseline availability check:
  - detects `aeon`, `sktime` and `ts2vec`;
  - marks fallback rows as not SOTA-claim eligible.
- DenseSensorJEPA MVP route:
  - temporal tokens;
  - temporal span masking;
  - EMA/shared target encoder;
  - position-conditioned predictor;
  - masked/visible/future latent losses;
  - avg/max/top-k token surprise scoring.

## Pending

- Official MiniROCKET/MultiROCKET/TS2Vec integration in the final benchmark
  protocol. In the current environment `aeon`, `sktime` and `ts2vec` are not
  installed.
- Token-level world model over DenseSensorJEPA tokens.
- Full CWRU/Paderborn representation benchmarks.
- Multi-seed, multi-horizon incremental runs beyond the quick MVP config.
- DenseSensorJEPA hard-split deltas after representative pretraining.

## Allowed Claims

- The CNC risk-scoring system can be useful as an MVP.
- Metadata/cycle are extremely strong CNC baselines.
- The repo now measures sensor/JEPA/world-model value as delta over metadata.
- DenseSensorJEPA is an implemented research path for local temporal surprise.

## Prohibited Claims Without More Evidence

- World model SOTA.
- Sensor-JEPA beats all strong baselines.
- DenseSensorJEPA learns reusable sensor representations.
- The model learns causal dynamics from action conditioning.
- Predictive-maintenance SOTA without official baselines, hard splits,
  multiple seeds and literature-comparable protocols.
