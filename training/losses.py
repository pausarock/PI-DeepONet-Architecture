"""Loss terms of the extended PI-DeepONet training framework.

L_total = α1 L_data + α2 L_phy
L_para  = L_fun + L_prev
L_ramp  = area-wise flux balance at user-supplied ramp sections

Weights default to 1.0. They are not experimental hyperparameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import torch

from models.pi_deeponet import PIDeepONet

from .datasets import DummyBatch
from .physics import equilibrium_speed, pw_residuals


@dataclass
class LossWeights:
    alpha_data: float = 1.0
    alpha_phy: float = 1.0
    lambda_v: float = 1.0
    lambda_q: float = 1.0
    lambda_prev: float = 1.0


def _require_grad(x: torch.Tensor) -> torch.Tensor:
    if x.requires_grad:
        return x
    return x.detach().clone().requires_grad_(True)


def data_loss(q_pred: torch.Tensor, v_pred: torch.Tensor, batch: DummyBatch) -> torch.Tensor:
    """Supervised mismatch at observed query points."""
    return torch.mean((q_pred - batch.q_obs) ** 2) + torch.mean((v_pred - batch.v_obs) ** 2)


def physics_loss(
    model: PIDeepONet,
    batch: DummyBatch,
    ramp_zones: Sequence[Tuple[float, float]],
    tau: float,
    c: float,
) -> torch.Tensor:
    """PW residuals at collocation points, with continuity masked on ramps."""
    x = _require_grad(batch.x)
    t = _require_grad(batch.t)
    q, v, density = model.operator(batch.u, x, t)
    v_f, k_c, _ = model.predict_parameters(batch.u, x)
    v_eq = equilibrium_speed(density, v_f, k_c)
    f1, f2 = pw_residuals(q, v, x, t, v_eq, ramp_zones, tau=tau, c=c)
    return torch.mean(f1 ** 2) + torch.mean(f2 ** 2)


def fundamental_diagram_loss(
    model: PIDeepONet,
    batch: DummyBatch,
    weights: LossWeights,
) -> torch.Tensor:
    """FD fitting: predicted states should match F(·; P)."""
    q, v, density = model.operator(batch.u, batch.x, batch.t)
    v_f, k_c, _ = model.predict_parameters(batch.u, batch.x)
    v_eq = equilibrium_speed(density.detach(), v_f, k_c)
    q_eq = v_eq * density.detach()
    return weights.lambda_v * torch.mean((v.detach() - v_eq) ** 2) + weights.lambda_q * torch.mean(
        (q.detach() - q_eq) ** 2
    )


def temporal_consistency_loss(
    model: PIDeepONet,
    batch_prev: DummyBatch,
    batch_curr: DummyBatch,
    weights: LossWeights,
) -> torch.Tensor:
    """P_prev(u_i) should stay close to P(u_{i-1})."""
    p_prev_est, _, _ = model.predict_parameters(batch_curr.u, batch_curr.x, component="prev")
    p_last, _, _ = model.predict_parameters(batch_prev.u, batch_prev.x, component="current")
    n = min(p_prev_est.shape[0], p_last.shape[0])
    if n <= 1:
        return torch.zeros((), device=batch_curr.u.device)
    return weights.lambda_prev * torch.mean((p_prev_est[1:n] - p_last[: n - 1].detach()) ** 2)


def ramp_conservation_loss(model: PIDeepONet, batch: DummyBatch) -> torch.Tensor:
    """Flux balance: predicted net accumulation vs. downstream − upstream flow.

    The public skeleton uses query-point flow as a stand-in for the area
    integral. Replace this with a spatial quadrature over each ramp zone.
    """
    q, _, _ = model.operator(batch.u, batch.x, batch.t)
    pred = q.reshape(batch.u.shape[0], -1).mean(dim=1)
    target = (batch.q_down - batch.q_up).detach()
    return torch.mean((pred - target) ** 2)


def compute_losses(
    model: PIDeepONet,
    batch_pre: DummyBatch,
    batch_res: DummyBatch,
    batch_para: DummyBatch,
    batch_para_prev: DummyBatch,
    batch_ramp: DummyBatch,
    weights: LossWeights,
    ramp_zones: Sequence[Tuple[float, float]],
    tau: float,
    c: float,
) -> Dict[str, torch.Tensor]:
    q, v, _ = model.operator(batch_pre.u, batch_pre.x, batch_pre.t)
    l_data = data_loss(q, v, batch_pre)
    l_phy = physics_loss(model, batch_res, ramp_zones, tau, c)
    l_fun = fundamental_diagram_loss(model, batch_para, weights)
    l_prev = temporal_consistency_loss(model, batch_para_prev, batch_para, weights)
    l_ramp = ramp_conservation_loss(model, batch_ramp)
    return {
        "data": l_data,
        "phy": l_phy,
        "total": weights.alpha_data * l_data + weights.alpha_phy * l_phy,
        "fun": l_fun,
        "prev": l_prev,
        "para": l_fun + l_prev,
        "ramp": l_ramp,
    }
