"""Attention-enhanced trunk with a nonlinear expansion of (x, t)."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class TrunkMLP(nn.Module):
    """Map a query point to a trunk feature vector.

    The 2-D coordinate `(x, t)` is expanded to eight elementary nonlinear
    features, then passed through an FCN with hidden-layer attention
    (U/V gates), following the extended PI-DeepONet trunk.

    Layer widths are placeholders. Spatial/temporal scaling is supplied by
    the caller; no corridor length is hard-coded.
    """

    def __init__(
        self,
        hidden_dims: Sequence[int] = (64, 64, 64),
        trunk_dim: int = 32,
    ) -> None:
        super().__init__()
        in_dim = 8
        hidden = list(hidden_dims)
        self.gate_u = nn.Linear(in_dim, hidden[0])
        self.gate_v = nn.Linear(in_dim, hidden[0])
        self.input_proj = nn.Linear(in_dim, hidden[0])
        self.hidden = nn.ModuleList(
            [nn.Linear(hidden[i], hidden[i + 1]) for i in range(len(hidden) - 1)]
        )
        self.head = nn.Linear(hidden[-1], trunk_dim)
        self.act = nn.Tanh()

    @staticmethod
    def expand_features(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = x.reshape(-1, 1)
        t = t.reshape(-1, 1)
        return torch.cat(
            [x, t, torch.exp(x), torch.exp(t), torch.cos(x), torch.cos(t), torch.sin(x), torch.sin(t)],
            dim=1,
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        z0 = self.expand_features(x, t)
        attn_u = self.act(self.gate_u(z0))
        attn_v = self.act(self.gate_v(z0))
        hidden = self.act(self.input_proj(z0))
        for layer in self.hidden:
            z = self.act(layer(hidden))
            hidden = (1.0 - z) * attn_u + z * attn_v
        return self.head(hidden)
