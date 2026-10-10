# Extended PI-DeepONet GTSP

Public skeleton of an **extended Physics-Informed Deep Operator Network** for freeway **global traffic state prediction (GTSP)**.

This repository follows the training framework of the extended PI-DeepONet paper: a CNN branch, an attention-enhanced trunk, a self-learning traffic-parameter predictor, a Payne–Whitham (PW) computational graph, and a ramp-mask / ramp-conservation mechanism.

**Training data, detector layouts, road geometry, calibrated hyperparameters, checkpoints, and site-specific scripts are omitted.** Layer widths and default tensor shapes are placeholders.

## Architecture

The operator maps a sparse historical window \(u\) (flow and speed) to traffic states at a query coordinate \(y=(x,t)\).

| Module | Role |
|---|---|
| CNN branch (CBN) | Dual-stream encoding of historical flow and speed fields |
| Attention trunk | Nonlinear feature expansion of \((x,t)\) plus hidden-layer attention |
| MIMO reconstruction | Inner product of branch and trunk features \(\rightarrow\) flow, speed, density |
| ParamCNN | Adaptive piecewise parameters \(P=(v_f,k_c,q_c)\) |
| PW graph | Continuity and momentum residuals for the physics loss |
| Ramp mask | Turns off pointwise conservation inside user-supplied ramp zones |
| Ramp conservation | Area-wise flux balance between upstream and downstream sections |

## Training framework

The loop follows **Algorithm 1** of the paper. Operator weights \(\theta\) (branch + trunk) and parameter-network weights \(\pi\) are updated on different mini-batches and at different intervals.

```
Input:  D_pre, D_res, D_para, D_ramp, D_val
        intervals Δ_para, Δ_ramp; patience N_P
Init:   θ, π randomly; L_best ← ∞

for t = 1 … N_iter:
    sample B_pre, B_res
    update θ by minimizing L_total = α1 L_data + α2 L_phy

    if t ≡ 0 (mod Δ_para):
        sample B_para
        update π by minimizing L_para = L_fun + L_prev

    if t ≡ 0 (mod Δ_ramp):
        sample B_ramp
        update θ by minimizing L_ramp

    L_val ← evaluate(D_val)
    early stop if no improvement for N_P evaluations
```

### Losses

- **\(L_\mathrm{data}\)** — supervised mismatch at observed query points (ICS / BCS / sensor locations).
- **\(L_\mathrm{phy}\)** — PW residuals \(f_1\) (continuity, ramp-masked) and \(f_2\) (momentum) at collocation points.
- **\(L_\mathrm{fun}\)** — fundamental-diagram consistency between predicted states and \(F(\cdot;P)\).
- **\(L_\mathrm{prev}\)** — temporal smoothness: \(P_\mathrm{prev}(u_i)\) should match \(P(u_{i-1})\).
- **\(L_\mathrm{ramp}\)** — integrated continuity residual over each ramp zone equals the observed (or reconstructed) upstream–downstream flux difference.

Loss weights \(\alpha_1,\alpha_2,\lambda_v,\lambda_q,\lambda_p\) and the update intervals are **user-supplied**. Defaults in this repo are unit placeholders, not experimental values.

## Quick start

```bash
pip install -r requirements.txt
python example_forward.py   # dummy operator forward pass
python example_train.py     # dummy Algorithm-1 loop on random tensors
```

Neither script loads traffic measurements.

## File layout

```
models/
  branch_cnn.py      # dual-stream CNN branch
  trunk_mlp.py       # nonlinear expansion + attention trunk
  param_cnn.py       # traffic-parameter predictor
  pi_deeponet.py     # operator assembly
training/
  physics.py         # PW residuals and ramp mask
  losses.py          # L_data, L_phy, L_fun, L_prev, L_ramp
  trainer.py         # Algorithm 1
  datasets.py        # batch protocol and dummy loaders
example_forward.py
example_train.py
```

## Citation

If you use this architecture, please cite the corresponding extended PI-DeepONet GTSP paper.
