import pandas as pd

from sensor_jepa.data.hard_splits import audit_identity_group_overlap, build_hard_split, hard_split_fingerprints


def test_hard_split_no_group_overlap():
    meta = pd.DataFrame({"ToolIndex": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]})
    split = build_hard_split(meta, "held_out_tool_id", seed=3)
    assert split.status == "ok"
    assert split.passes_no_overlap
    assert not (set(split.train_groups) & set(split.test_groups))
    assert split.train_mask.sum() > 0
    assert split.test_mask.sum() > 0


def test_hard_split_missing_column_pending():
    meta = pd.DataFrame({"other": [1, 2, 3]})
    split = build_hard_split(meta, "held_out_tool_id", seed=3)
    assert split.status == "pending"
    assert "missing" in split.reason


def test_hardness_split_keeps_equal_physical_values_together():
    # The repeated boundary values used to be separated by rank(method="first").
    meta = pd.DataFrame(
        {
            "HardnessMean": [38.0, 38.33, 38.33, 38.33, 39.0, 39.33, 39.33, 40.0, 41.0, 42.0] * 2,
        }
    )
    split = build_hard_split(meta, "held_out_hardness_bin", seed=7)
    assert split.status == "ok"
    assert split.passes_no_overlap
    assert split.passes_physical_no_overlap
    train = set(meta.loc[split.train_mask, "HardnessMean"])
    val = set(meta.loc[split.val_mask, "HardnessMean"])
    test = set(meta.loc[split.test_mask, "HardnessMean"])
    assert not (train & val or train & test or val & test)
    assert split.physical_column == "HardnessMean"
    first = hard_split_fingerprints(split)
    repeated = hard_split_fingerprints(build_hard_split(meta, "held_out_hardness_bin", seed=7))
    assert first == repeated
    assert all(len(value) == 64 for value in first.values())


def test_hardness_split_keeps_linked_tools_and_parts_together():
    meta = pd.DataFrame(
        {
            "HardnessMean": [35, 36, 36, 37, 38, 39],
            "ToolIndex": [1, 1, 2, 2, 3, 4],
            "FileName": ["P001_F01", "P001_F02", "P002_F01", "P002_F02", "P003_F01", "P004_F01"],
        }
    )
    split = build_hard_split(meta, "held_out_hardness_bin", seed=7)
    assert split.status == "ok"
    assert split.passes_physical_no_overlap
    audits = audit_identity_group_overlap(meta, split)
    assert audits
    assert all(entry["passes_no_overlap"] for entry in audits.values())
    assert all(len(entry["sha256"]) == 64 for entry in audits.values())
