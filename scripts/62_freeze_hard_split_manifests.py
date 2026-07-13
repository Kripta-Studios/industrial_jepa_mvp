from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import _bootstrap  # noqa: F401

from common.config import load_config
from sensor_jepa.data.cnc_world_model import prepare_transition_from_config
from sensor_jepa.data.hard_splits import audit_identity_group_overlap, build_hard_split, hard_split_report_rows
from sensor_jepa.train.hard_generalization import _concat_bundle


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(payload: object) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _coarse_attribute_audit(metadata, split) -> dict[str, object]:
    """Disclose coarse covariates that cannot serve as physical identity IDs."""

    audits: dict[str, object] = {}
    for column in ["MillingToolType"]:
        if column not in metadata:
            continue
        values = metadata[column].astype("string").fillna("missing").astype(str)
        parts = {
            "train": sorted(values[split.train_mask].unique().tolist()),
            "validation": sorted(values[split.val_mask].unique().tolist()),
            "test": sorted(values[split.test_mask].unique().tolist()),
        }
        tr, va, te = map(set, parts.values())
        audits[column] = {
            **parts,
            "intersections": {
                "train_validation": sorted(tr & va),
                "train_test": sorted(tr & te),
                "validation_test": sorted(va & te),
            },
            "included_in_identity_component_graph": False,
            "reason": "coarse process category, not a physical identity; including it connects otherwise independent tools and prevents a useful three-way partition",
        }
    return audits


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze hard-split raw-value memberships and hashes.")
    parser.add_argument("--config", default="configs/sensor_jepa/hard_generalization_cnc.yaml")
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--model-seeds", default="42,123,999")
    parser.add_argument("--split", default="held_out_hardness_bin")
    parser.add_argument("--output", default="artifacts/splits/hardness_raw_split_manifest.json")
    args = parser.parse_args()

    config_path = Path(args.config)
    cfg = load_config(config_path)
    raw_root = Path(cfg["data"]["raw_root"])
    sources = [raw_root / "FeatureAndMetadata_Milling.csv", raw_root / "metadata.xlsx"]
    source_hashes = {str(path): _sha256_file(path) for path in sources if path.exists()}
    seeded = deepcopy(cfg)
    seeded["seed"] = args.data_seed
    bundle = prepare_transition_from_config(seeded)
    _, _, metadata = _concat_bundle(bundle)
    split = build_hard_split(metadata, args.split, seed=args.split_seed)
    row = hard_split_report_rows([split])[0]
    identity_audit = audit_identity_group_overlap(metadata, split)
    identity_columns = [column for column in ["FileName", "source_cycle", "ToolIndex", split.physical_column] if column in metadata]
    identities = metadata[identity_columns].where(metadata[identity_columns].notna(), None).to_dict("records")
    frozen_split = {
        "data_seed": args.data_seed,
        "split_seed": args.split_seed,
        "model_seeds": [int(value.strip()) for value in args.model_seeds.split(",") if value.strip()],
        "n_rows": len(metadata),
        "ordered_sample_identity_sha256": _sha256_json(identities),
        "identity_group_audit": identity_audit,
        "coarse_attribute_audit": _coarse_attribute_audit(metadata, split),
        "passes_all_identity_group_no_overlap": all(audit["passes_no_overlap"] for audit in identity_audit.values()),
        **row,
    }

    payload = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_post_fix_split_definition",
        "config_path": str(config_path),
        "config_sha256": _sha256_file(config_path),
        "source_sha256": source_hashes,
        "split_name": args.split,
        "frozen_split": frozen_split,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
