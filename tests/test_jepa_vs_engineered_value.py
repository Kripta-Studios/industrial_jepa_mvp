import pandas as pd

from sensor_jepa.train.jepa_vs_engineered_value import _add_engineered_deltas


def test_engineered_deltas_match_protocol_seed_horizon_and_target():
    df = pd.DataFrame(
        [
            {
                "protocol": "no_cycle",
                "split_name": "no_cycle",
                "seed": 42,
                "forecast_horizon": 1,
                "failure_horizon_cycles": 5,
                "model_name": "sensor_engineered_only",
                "AUPRC": 0.40,
                "precision_at_10pct": 0.20,
                "recall_at_10pct": 0.30,
                "mean_lead_time": 2.0,
            },
            {
                "protocol": "no_cycle",
                "split_name": "no_cycle",
                "seed": 42,
                "forecast_horizon": 3,
                "failure_horizon_cycles": 5,
                "model_name": "sensor_engineered_only",
                "AUPRC": 0.90,
                "precision_at_10pct": 0.80,
                "recall_at_10pct": 0.70,
                "mean_lead_time": 9.0,
            },
            {
                "protocol": "no_cycle",
                "split_name": "no_cycle",
                "seed": 42,
                "forecast_horizon": 1,
                "failure_horizon_cycles": 5,
                "model_name": "metadata_plus_sensor_engineered",
                "AUPRC": 0.45,
                "precision_at_10pct": 0.20,
                "recall_at_10pct": 0.30,
                "mean_lead_time": 2.0,
            },
            {
                "protocol": "no_cycle",
                "split_name": "no_cycle",
                "seed": 42,
                "forecast_horizon": 1,
                "failure_horizon_cycles": 5,
                "model_name": "sensor_engineered_plus_current_z",
                "AUPRC": 0.52,
                "precision_at_10pct": 0.25,
                "recall_at_10pct": 0.45,
                "mean_lead_time": 3.5,
            },
        ]
    )
    out = _add_engineered_deltas(df)
    row = out[out["model_name"].eq("sensor_engineered_plus_current_z")].iloc[0]
    assert abs(row["delta_AUPRC_vs_sensor_engineered"] - 0.12) < 1e-9
    assert abs(row["delta_Precision10_vs_sensor_engineered"] - 0.05) < 1e-9
    assert abs(row["delta_Recall10_vs_sensor_engineered"] - 0.15) < 1e-9
    assert abs(row["delta_lead_time_vs_sensor_engineered"] - 1.5) < 1e-9
    assert abs(row["delta_AUPRC_vs_metadata_plus_sensor_engineered"] - 0.07) < 1e-9
