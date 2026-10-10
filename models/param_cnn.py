"""1-D CNN that predicts piecewise traffic-flow parameters from the input window."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class ParamCNN(nn.Module):
    """Map `u` of shape `(B, T, N * 2)` to a parameter vector.

    The intended output layout is, for each road segment:
    `[prev_vf, prev_kc, prev_cap, d_vf, d_kc, d_cap]`.
    Convolution and MLP widths are placeholders.
    """

    def __init__(
        self,
        history_len: int,
        num_sensors: int,
        output_dim: int,
        conv_channels: Sequence[int] = (32, 64, 64),
        mlp_hidden: int = 128,
    ) -> None:
        super().__init__()
        self.history_len = history_len
        self.num_sensors = num_sensors
        in_ch = num_sensors * 2

        convs = []
        prev = in_ch
        for ch in conv_channels:
            convs.extend([nn.Conv1d(prev, ch, kernel_size=3, padding=1), nn.ReLU()])
            prev = ch
        self.conv = nn.Sequential(*convs)
        self.head = nn.Sequential(
            nn.Linear(prev * history_len, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, output_dim),
        )

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        # (B, T, F) -> (B, F, T) for Conv1d over time
        x = u.permute(0, 2, 1)
        x = self.conv(x).flatten(1)
        return self.head(x)
