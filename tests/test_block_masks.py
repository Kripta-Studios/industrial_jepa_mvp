import torch

from visual_jepa.data.block_masks import BlockMaskGenerator


def test_block_masks_reproducible_and_non_empty():
    gen_a = BlockMaskGenerator(mask_ratio=0.5, num_target_blocks=3, min_block_size=1, max_block_size=2, seed=123)
    gen_b = BlockMaskGenerator(mask_ratio=0.5, num_target_blocks=3, min_block_size=1, max_block_size=2, seed=123)

    a = gen_a(batch_size=2, grid_shape=(4, 4))
    b = gen_b(batch_size=2, grid_shape=(4, 4))

    assert torch.equal(a.target_mask, b.target_mask)
    assert torch.equal(a.visible_mask, b.visible_mask)
    assert a.target_mask.any(dim=1).all()
    assert a.visible_mask.any(dim=1).all()
    assert a.target_indices.shape[0] == 2
    assert a.visible_indices.shape[0] == 2
    ratio = a.target_mask.float().mean().item()
    assert 0.25 <= ratio <= 0.75
    assert len(a.target_blocks) == 2

