import pandas as pd

from industrial_world_model.hierarchy.aggregation import aggregate_scores, group_risk_table, top_alerts


def test_patch_to_group_aggregation_and_ranking():
    assert aggregate_scores([1, 5, 2], method="max") == 5
    df = pd.DataFrame({"cycle": ["a", "a", "b"], "risk_score": [0.2, 0.8, 0.5]})
    grouped = group_risk_table(df, "cycle", "risk_score", method="topk_mean")
    assert grouped.iloc[0]["group"] == "a"
    alerts = top_alerts(df, k=2)
    assert list(alerts["risk_score"]) == [0.8, 0.5]
