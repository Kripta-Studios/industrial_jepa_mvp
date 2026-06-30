from __future__ import annotations

import torch


def synthetic_latent_sequence(n: int = 128, latent_dim: int = 32, action_dim: int = 8, seed: int = 42):
    gen = torch.Generator().manual_seed(seed)
    z = torch.randn(n, latent_dim, generator=gen)
    a = torch.randn(n, action_dim, generator=gen)
    z_next = z + 0.2 * torch.tanh(a[:, :1]) + 0.05 * torch.randn(n, latent_dim, generator=gen)
    return z, a, z_next
