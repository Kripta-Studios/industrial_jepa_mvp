import torch

from visual_jepa.data.dense_patching import grid_positions, patch_mask_to_image, patchify, unpatchify


def test_patchify_unpatchify_round_trip():
    x = torch.arange(1 * 2 * 8 * 8, dtype=torch.float32).reshape(1, 2, 8, 8)
    patches, grid = patchify(x, patch_size=4)
    assert grid == (2, 2)
    assert patches.shape == (1, 4, 2 * 4 * 4)

    recon = unpatchify(patches, grid, patch_size=4, channels=2)
    assert recon.shape == x.shape
    assert torch.equal(recon, x)


def test_patch_mask_to_image_and_positions():
    mask = torch.tensor([[True, False, False, True]])
    image_mask = patch_mask_to_image(mask, (2, 2), patch_size=4)
    assert image_mask.shape == (1, 1, 8, 8)
    assert image_mask[:, :, :4, :4].all()
    assert image_mask[:, :, 4:, 4:].all()

    pos = grid_positions((2, 3))
    assert pos.shape == (6, 2)
    assert float(pos.min()) >= -1.0
    assert float(pos.max()) <= 1.0

