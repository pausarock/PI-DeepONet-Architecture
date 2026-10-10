"""Dual-stream CNN branch that encodes historical flow and speed fields."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn


class BranchCNN(nn.Module):
    """Encode a sensor window `u` of shape `(B, T, N * 2)` into two branch vectors.

    The last feature axis is assumed to interleave flow and speed per sensor.
    Convolution widths and MLP widths are placeholders.
    """

    def __init__(
        self,
        history_len: int,
        num_sensors: int,
        hidden_dims: Sequence[int] = (64, 64),
        branch_dim: int = 32,
        conv_channels: Sequence[int] = (8, 8),
    ) -> None:
        super().__init__()
        self.history_len = history_len
        self.num_sensors = num_sensors
        self.branch_dim = branch_dim

        self.flow_conv = self._make_conv(conv_channels)
        self.speed_conv = self._make_conv(conv_channels)

        dummy = torch.zeros(1, 1, history_len, num_sensors)
        flat_dim = self.flow_conv(dummy).shape[1]
        mlp_dims = [flat_dim, *hidden_dims, branch_dim]
        self.flow_mlp = self._make_mlp(mlp_dims)
        self.speed_mlp = self._make_mlp(mlp_dims)

    @staticmethod
    def _make_conv(channels: Sequence[int]) -> nn.Sequential:
        layers = []
        in_ch = 1
        for out_ch in channels:
            layers.extend(
                [
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                    nn.Tanh(),
                ]
            )
            in_ch = out_ch
        layers.append(nn.Flatten())
        return nn.Sequential(*layers)

    @staticmethod
    def _make_mlp(dims: Sequence[int]) -> nn.Sequential:
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.Tanh())
        return nn.Sequential(*layers)

    def forward(self, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = u.shape[0]
        x = u.reshape(batch, self.history_len, self.num_sensors, 2).permute(0, 3, 1, 2)
        flow = self.flow_mlp(self.flow_conv(x[:, 0:1]))
        speed = self.speed_mlp(self.speed_conv(x[:, 1:2]))
        return flow, speed
