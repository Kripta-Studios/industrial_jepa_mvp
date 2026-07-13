# Results invalidation notice

Effective 2026-07-13, every performance claim produced before the corrected
raw-hardness split and validation-only threshold protocol is **invalidated**.
This includes tables, PDFs, HTML decks, demo payloads, and files under legacy
`outputs/` or `results/` paths. They may be retained only as historical
artifacts and must not be cited as evidence.

The invalidation has two independent causes:

1. hardness was previously split at row level, allowing equal physical values
   to occur in more than one partition;
2. some visual anomaly metrics selected a decision threshold using test labels.

An artifact is claim-eligible only if its registry entry has status
`claim_eligible`, names an immutable configuration and data/split hashes,
records every seed, was produced from a clean commit, and passed the gates in
`RERUN_PLAN.md`. There are currently **no claim-eligible post-audit results**.

The run in `outputs/post_audit/hard_generalization_20260713/` is deliberately
registered as invalid evidence: it was executed while corrections were
uncommitted, allowed tool IDs to cross partitions, and conflated split seeds
with model seeds. Its numbers cannot repair or replace the invalidated claims.
