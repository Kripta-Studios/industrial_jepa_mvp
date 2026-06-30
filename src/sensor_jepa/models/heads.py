from __future__ import annotations

from torch import nn


class ClassificationHead(nn.Module):
    def __init__(self, embedding_dim: int, num_classes: int, hidden_dim: int | None = None):
        super().__init__()
        if hidden_dim:
            self.net = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            self.net = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        return self.net(x)

