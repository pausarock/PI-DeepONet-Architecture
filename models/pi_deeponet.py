"""Assemble branch, trunk, and parameter networks into a PI-DeepONet operator."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .branch_cnn import BranchCNN
from .param_cnn import ParamCNN
from .trunk_mlp import TrunkMLP


class PIDeepONet(nn.Module):
    """Physics-informed DeepONet skeleton.

    Query the operator at coordinates `(x, t)` given a historical window `u`.
    Road-segment dictionaries are supplied by the user; this file does not
    contain any real corridor geometry or calibrated ranges.
    """

    def __init__(
        self,
        history_len: int,
        num_sensors: int,
        num_segments: int,
        branch_hidden: Sequence[int] = (64, 64),
        trunk_hidden: Sequence[int] = (64, 64, 64),
        feature_dim: int = 32,
        flow_scale: float = 1.0,
        speed_scale: float = 1.0,
        road_segments: Optional[List[Dict]] = None,
    ) -> None:
        super().__init__()
        self.history_len = history_len
        self.num_sensors = num_sensors
        self.flow_scale = flow_scale
        self.speed_scale = speed_scale
        self.road_segments = road_segments or _placeholder_segments(num_segments)

        self.branch = BranchCNN(
            history_len=history_len,
            num_sensors=num_sensors,
            hidden_dims=branch_hidden,
            branch_dim=2 * feature_dim,
        )
        self.trunk = TrunkMLP(hidden_dims=trunk_hidden, trunk_dim=feature_dim)
        self.param_net = ParamCNN(
            history_len=history_len,
            num_sensors=num_sensors,
            output_dim=6 * len(self.road_segments),
        )

    def operator(self, u: torch.Tensor, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return `(flow, speed, density)` at query locations.

        `u`: `(B, T, N * 2)`
        `x`, `t`: `(Q,)` query coordinates; `u` is repeated if `Q != B`.
        """
        b_flow, b_speed = self.branch(u)
        trunk = self.trunk(x, t)
        if b_flow.shape[0] != trunk.shape[0]:
            repeat = trunk.shape[0] // b_flow.shape[0]
            b_flow = b_flow.repeat_interleave(repeat, dim=0)
            b_speed = b_speed.repeat_interleave(repeat, dim=0)

        half = trunk.shape[1]
        q = torch.sigmoid((b_flow[:, :half] * trunk).sum(dim=1) + (b_speed[:, :half] * trunk).sum(dim=1))
        v = torch.sigmoid((b_flow[:, half:] * trunk).sum(dim=1) + (b_speed[:, half:] * trunk).sum(dim=1))
        q = q * self.flow_scale
        v = v * self.speed_scale + 1e-6
        return q, v, q / v

    def predict_parameters(self, u: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return piecewise `(v_f, k_c, capacity)` at locations `x`.

        Parameter bounds must be provided through `road_segments` by the user.
        Default bounds below are generic placeholders, not calibrated values.
        """
        raw = self.param_net(u)
        n_seg = len(self.road_segments)
        prev = torch.sigmoid(raw[:, : 3 * n_seg])
        delta = torch.tanh(raw[:, 3 * n_seg :])
        base = torch.sigmoid((prev + 0.1 * delta - 0.5) * 10.0)

        v_f = torch.zeros_like(x.reshape(-1))
        k_c = torch.zeros_like(v_f)
        cap = torch.zeros_like(v_f)
        for i, seg in enumerate(self.road_segments):
            mask = (x.reshape(-1) >= seg["start"]) & (x.reshape(-1) <= seg["end"])
            vf0, vf1 = seg.get("v_f_range", (0.0, 1.0))
            kc0, kc1 = seg.get("k_c_range", (0.0, 1.0))
            c0, c1 = seg.get("capacity_range", (0.0, 1.0))
            v_f = torch.where(mask, base[:, i * 3] * (vf1 - vf0) + vf0, v_f)
            k_c = torch.where(mask, base[:, i * 3 + 1] * (kc1 - kc0) + kc0, k_c)
            cap = torch.where(mask, base[:, i * 3 + 2] * (c1 - c0) + c0, cap)
        return v_f, k_c, cap


def _placeholder_segments(num_segments: int) -> List[Dict]:
    segs = []
    for i in range(num_segments):
        segs.append(
            {
                "start": float(i),
                "end": float(i + 1),
                "lanes": 1,
                "v_f_range": (0.0, 1.0),
                "k_c_range": (0.0, 1.0),
                "capacity_range": (0.0, 1.0),
            }
        )
    return segs
