"""Payne–Whitham residuals and the ramp mask.

Governing equations (user supplies τ, c, and the equilibrium map F):

    f1 = ∂(q / v) / ∂t + ∂q / ∂x          (continuity)
    f2 = ∂v/∂t + v ∂v/∂x + (c v / q) ∂(q/v)/∂x + (v - F)/τ

Inside ramp influence zones the pointwise continuity residual is masked:
    f1_hat = M(x) * f1,   M = 0 on ramps, M = 1 elsewhere.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import torch


def equilibrium_speed(
    density: torch.Tensor,
    v_f: torch.Tensor,
    k_c: torch.Tensor,
) -> torch.Tensor:
    """Placeholder fundamental diagram. Replace with the FD used in training."""
    k_c = k_c.clamp(min=1e-6)
    return v_f * torch.exp(-(density / k_c))


def ramp_mask(x: torch.Tensor, ramp_zones: Sequence[Tuple[float, float]]) -> torch.Tensor:
    """Return 1 outside ramp zones and 0 inside them."""
    mask = torch.ones_like(x)
    for start, end in ramp_zones:
        inside = (x >= start) & (x <= end)
        mask = torch.where(inside, torch.zeros_like(mask), mask)
    return mask


def pw_residuals(
    q: torch.Tensor,
    v: torch.Tensor,
    x: torch.Tensor,
    t: torch.Tensor,
    v_eq: torch.Tensor,
    ramp_zones: Sequence[Tuple[float, float]],
    tau: float = 1.0,
    c: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return masked continuity residual `f1` and momentum residual `f2`.

    `tau` and `c` default to unit placeholders. Set them to the PW constants
    used in a given study; this file does not hard-code experimental values.
    """
    density = q / v.clamp(min=1e-6)
    d_t = torch.autograd.grad(density.sum(), t, create_graph=True, retain_graph=True)[0]
    q_x = torch.autograd.grad(q.sum(), x, create_graph=True, retain_graph=True)[0]
    d_x = torch.autograd.grad(density.sum(), x, create_graph=True, retain_graph=True)[0]
    v_x = torch.autograd.grad(v.sum(), x, create_graph=True, retain_graph=True)[0]
    v_t = torch.autograd.grad(v.sum(), t, create_graph=True, retain_graph=True)[0]

    f1 = d_t + q_x
    f1 = f1 * ramp_mask(x, ramp_zones)
    f2 = v_t + v * v_x + (c * v / q.clamp(min=1e-6)) * d_x + (v - v_eq) / max(tau, 1e-6)
    return f1, f2
