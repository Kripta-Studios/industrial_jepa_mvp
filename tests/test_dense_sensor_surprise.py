import torch

from sensor_jepa.eval.dense_sensor_surprise import surprise_scores_from_errors


def test_dense_sensor_surprise_shapes():
    errors = torch.tensor([[0.1, 0.4, 0.2, 0.3], [0.0, 1.0, 0.5, 0.2]])
    scores = surprise_scores_from_errors(errors, topk_ratio=0.5)
    assert scores["avg_surprise"].shape == (2,)
    assert scores["max_surprise"].shape == (2,)
    assert scores["topk_surprise"].shape == (2,)
    assert scores["ewma_surprise"].shape == (2,)
    assert scores["surprise_slope"].shape == (2,)
    assert scores["surprise_q90"].shape == (2,)
    assert scores["surprise_persistence"].shape == (2,)
    assert scores["temporal_surprise_curve"].shape == (2, 4)
    assert torch.all(scores["max_surprise"] >= scores["avg_surprise"])
