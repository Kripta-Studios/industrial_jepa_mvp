import numpy as np
import pytest

from industrial_world_model.visual.eval import fit_threshold, image_anomaly_metrics


def test_test_labels_cannot_change_frozen_validation_threshold():
    val_scores = np.array([0.1, 0.2, 0.3, 0.4])
    threshold = fit_threshold(None, val_scores, method="nominal_quantile", nominal_quantile=0.75)
    scores = np.array([0.2, 0.8, 0.7, 0.1])
    first = image_anomaly_metrics(
        np.array([0, 1, 1, 0]), scores, threshold=threshold, threshold_source="train_nominal_validation"
    )
    mutated = image_anomaly_metrics(
        np.array([1, 0, 0, 1]), scores, threshold=threshold, threshold_source="train_nominal_validation"
    )
    assert first["threshold"] == mutated["threshold"] == threshold
    assert first["threshold_source"] == "train_nominal_validation"


def test_best_f1_rejects_single_class_validation():
    with pytest.raises(ValueError, match="both classes"):
        fit_threshold(np.zeros(3), np.array([0.1, 0.2, 0.3]), method="best_f1")


def test_threshold_dependent_metrics_are_absent_without_frozen_threshold():
    result = image_anomaly_metrics(np.array([0, 1]), np.array([0.1, 0.9]))
    assert result["image_AUROC"] == 1.0
    assert result["F1"] is None
    assert result["balanced_accuracy"] is None
    assert result["threshold_source"] == "not_provided"
