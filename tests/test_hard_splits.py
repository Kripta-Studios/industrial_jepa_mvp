import pandas as pd

from sensor_jepa.data.hard_splits import build_hard_split


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
