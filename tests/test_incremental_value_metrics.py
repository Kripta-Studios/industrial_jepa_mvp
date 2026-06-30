import pandas as pd

from sensor_jepa.eval.incremental_metrics import add_delta_vs_metadata


def test_delta_vs_metadata_uses_matching_horizon_and_target():
    df = pd.DataFrame(
        [
            {"protocol": "operational", "seed": 1, "forecast_horizon": 1, "failure_horizon_cycles": 5, "model_name": "metadata_only", "AUPRC": 0.5, "AUROC": 0.6, "precision_at_10pct": 0.2, "recall_at_10pct": 0.3},
            {"protocol": "operational", "seed": 1, "forecast_horizon": 2, "failure_horizon_cycles": 5, "model_name": "metadata_only", "AUPRC": 0.8, "AUROC": 0.9, "precision_at_10pct": 0.4, "recall_at_10pct": 0.5},
            {"protocol": "operational", "seed": 1, "forecast_horizon": 1, "failure_horizon_cycles": 5, "model_name": "metadata_plus_current_z", "AUPRC": 0.7, "AUROC": 0.65, "precision_at_10pct": 0.25, "recall_at_10pct": 0.35},
        ]
    )
    out = add_delta_vs_metadata(df)
    row = out[out["model_name"].eq("metadata_plus_current_z")].iloc[0]
    assert abs(row["delta_AUPRC_vs_metadata_only"] - 0.2) < 1e-9
    assert abs(row["delta_AUROC_vs_metadata_only"] - 0.05) < 1e-9
    assert abs(row["delta_Precision@10_vs_metadata_only"] - 0.05) < 1e-9
