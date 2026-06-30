import torch

from sensor_jepa.data.temporal_masks import TemporalMaskGenerator


def test_temporal_masks_non_empty_and_reproducible():
    gen1 = TemporalMaskGenerator(num_tokens=10, temporal_mask_ratio=0.4, seed=7)
    gen2 = TemporalMaskGenerator(num_tokens=10, temporal_mask_ratio=0.4, seed=7)
    m1 = gen1(batch_size=2)
    m2 = gen2(batch_size=2)
    assert m1["target_mask"].any()
    assert m1["context_mask"].any()
    assert torch.equal(m1["target_mask"], m2["target_mask"])
    ratio = m1["target_mask"][0].float().mean().item()
    assert 0.1 <= ratio <= 0.8
