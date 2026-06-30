import torch

from sensor_jepa.data.temporal_tokenization import TemporalPatchTokenizer, count_temporal_tokens, temporal_patchify


def test_temporal_patchify_shapes():
    x = torch.arange(2 * 8 * 3, dtype=torch.float32).view(2, 8, 3)
    patches = temporal_patchify(x, patch_size=2, stride=2)
    assert patches.shape == (2, 4, 6)
    assert count_temporal_tokens(8, 2, 2) == 4


def test_temporal_tokenizer_outputs_embeddings():
    x = torch.randn(2, 8, 3)
    tokenizer = TemporalPatchTokenizer(input_channels=3, embedding_dim=16, temporal_patch_size=2, temporal_patch_stride=2)
    tokens = tokenizer(x)
    assert tokens.shape == (2, 4, 16)
