"""Batch protocol for the four training sets in Algorithm 1.

Callers replace `DummyLoaders` with their own data. This file does not
store measurements, detector IDs, or corridor geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Tuple

import torch


@dataclass
class DummyBatch:
    """One mini-batch for operator / physics / parameter / ramp updates.

    `u`: historical window, shape `(B, T, N * 2)`
    `x`, `t`: query coordinates, shape `(Q,)`
    `q_obs`, `v_obs`: optional observations at the query points
    `q_up`, `q_down`: optional ramp-boundary flows
    """

    u: torch.Tensor
    x: torch.Tensor
    t: torch.Tensor
    q_obs: torch.Tensor
    v_obs: torch.Tensor
    q_up: torch.Tensor
    q_down: torch.Tensor


def make_dummy_batch(
    batch_size: int,
    history_len: int,
    num_sensors: int,
    num_segments: int,
) -> DummyBatch:
    """Random tensors used only to exercise the training loop."""
    q = batch_size
    return DummyBatch(
        u=torch.randn(batch_size, history_len, num_sensors * 2),
        x=torch.rand(q) * float(num_segments),
        t=torch.rand(q),
        q_obs=torch.rand(q),
        v_obs=torch.rand(q) + 0.1,
        q_up=torch.rand(batch_size),
        q_down=torch.rand(batch_size),
    )


class DummyLoaders:
    """Infinite dummy streams standing in for D_pre / D_res / D_para / D_ramp / D_val."""

    def __init__(
        self,
        history_len: int,
        num_sensors: int,
        num_segments: int,
        batch_size: int = 4,
    ) -> None:
        self.history_len = history_len
        self.num_sensors = num_sensors
        self.num_segments = num_segments
        self.batch_size = batch_size

    def _one(self) -> DummyBatch:
        return make_dummy_batch(
            self.batch_size, self.history_len, self.num_sensors, self.num_segments
        )

    def stream(self) -> Iterator[DummyBatch]:
        while True:
            yield self._one()

    def pre(self) -> DummyBatch:
        return self._one()

    def res(self) -> DummyBatch:
        return self._one()

    def para(self) -> DummyBatch:
        return self._one()

    def ramp(self) -> DummyBatch:
        return self._one()

    def val(self) -> DummyBatch:
        return self._one()
