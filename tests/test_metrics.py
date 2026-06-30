import numpy as np

from common.metrics import anomaly_metrics, classification_metrics, regression_metrics


def test_classification_metrics():
    m = classification_metrics(np.array([0, 1, 1]), np.array([0, 1, 0]))
    assert "macro_F1" in m
    assert m["accuracy"] > 0


def test_regression_metrics():
    m = regression_metrics(np.array([1.0, 2.0]), np.array([1.0, 2.5]))
    assert m["RMSE"] >= 0


def test_anomaly_metrics():
    m = anomaly_metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))
    assert m["AUROC"] == 1.0

