"""Algorithm 1: training procedure for extended PI-DeepONet GTSP.

Operator parameters θ (branch + trunk) and parameter-network weights π
are updated on separate schedules. Interval / patience / iteration counts
below are placeholders.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn

from models.pi_deeponet import PIDeepONet

from .datasets import DummyLoaders
from .losses import LossWeights, compute_losses, data_loss


@dataclass
class TrainConfig:
    n_iter: int = 4
    delta_para: int = 1
    delta_ramp: int = 2
    patience: int = 10
    lr_operator: float = 1e-3
    lr_param: float = 1e-3
    tau: float = 1.0
    c: float = 1.0
    ramp_zones: Sequence[Tuple[float, float]] = ()
    weights: LossWeights = field(default_factory=LossWeights)


def _operator_params(model: PIDeepONet) -> List[nn.Parameter]:
    return list(model.branch.parameters()) + list(model.trunk.parameters())


def _param_params(model: PIDeepONet) -> List[nn.Parameter]:
    return list(model.param_net.parameters())


def evaluate(model: PIDeepONet, loaders: DummyLoaders) -> float:
    """Validation proxy: supervised error on a dummy observed batch."""
    model.eval()
    batch = loaders.val()
    with torch.no_grad():
        q, v, _ = model.operator(batch.u, batch.x, batch.t)
        value = float(data_loss(q, v, batch).item())
    model.train()
    return value


def train_loop(
    model: PIDeepONet,
    loaders: DummyLoaders,
    config: TrainConfig,
) -> Dict[str, List[float]]:
    """Run Algorithm 1. Replace `loaders` with real datasets before training."""
    opt_theta = torch.optim.Adam(_operator_params(model), lr=config.lr_operator)
    opt_pi = torch.optim.Adam(_param_params(model), lr=config.lr_param)

    history: Dict[str, List[float]] = {k: [] for k in ("total", "para", "ramp", "val")}
    best = float("inf")
    stall = 0
    batch_para_prev = loaders.para()

    model.train()
    for step in range(1, config.n_iter + 1):
        batch_pre = loaders.pre()
        batch_res = loaders.res()
        batch_para = loaders.para()
        batch_ramp = loaders.ramp()

        losses = compute_losses(
            model,
            batch_pre,
            batch_res,
            batch_para,
            batch_para_prev,
            batch_ramp,
            config.weights,
            config.ramp_zones,
            config.tau,
            config.c,
        )

        opt_theta.zero_grad()
        losses["total"].backward()
        opt_theta.step()

        if step % config.delta_para == 0:
            losses = compute_losses(
                model,
                batch_pre,
                batch_res,
                batch_para,
                batch_para_prev,
                batch_ramp,
                config.weights,
                config.ramp_zones,
                config.tau,
                config.c,
            )
            opt_pi.zero_grad()
            losses["para"].backward()
            opt_pi.step()

        if step % config.delta_ramp == 0:
            losses = compute_losses(
                model,
                batch_pre,
                batch_res,
                batch_para,
                batch_para_prev,
                batch_ramp,
                config.weights,
                config.ramp_zones,
                config.tau,
                config.c,
            )
            opt_theta.zero_grad()
            losses["ramp"].backward()
            opt_theta.step()

        val = evaluate(model, loaders)
        history["total"].append(float(losses["total"].detach()))
        history["para"].append(float(losses["para"].detach()))
        history["ramp"].append(float(losses["ramp"].detach()))
        history["val"].append(val)

        if val < best:
            best = val
            stall = 0
        else:
            stall += 1
            if stall > config.patience:
                break

        batch_para_prev = batch_para

    return history
