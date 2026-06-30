import torch

from visual_jepa.data.patching import random_block_mask_images


def test_random_block_mask_images():
    x = torch.ones(2, 3, 32, 32)
    mask = random_block_mask_images(x, patch_size=8, mask_ratio=0.5)
    assert mask.shape == (2, 1, 32, 32)
    assert mask.any()
    assert mask.float().mean() > 0.2

