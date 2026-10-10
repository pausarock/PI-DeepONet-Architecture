"""Trunk MLP that evaluates the operator at a query coordinate (x, t)."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class TrunkMLP(nn.Module):
    """Map a query point to a trunk feature vector.

    The 2-D coordinate `(x, t)` is first expanded by elementary nonlinear
    features, then passed through an MLP. Layer widths are placeholders.
    Spatial/temporal scaling should be supplied by the caller; no site length
    is hard-coded here.
    """

    def __init__(
        self,
        hidden_dims: Sequence[int] = (64, 64, 64),
        trunk_dim: int = 32,
    ) -> None:
        super().__init__()
        dims = [8, *hidden_dims, trunk_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    @staticmethod
    def expand_features(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = x.reshape(-1, 1)
        t = t.reshape(-1, 1)
        return torch.cat(
            [x, t, torch.exp(x), torch.exp(t), torch.cos(x), torch.cos(t), torch.sin(x), torch.sin(t)],
            dim=1,
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(self.expand_features(x, t))
