# PI-DeepONet Network Architecture (Public Skeleton)

This repository contains **only the network-structure skeleton** of a Physics-Informed Deep Operator Network (PI-DeepONet) for freeway global traffic state prediction.

It is intended for architecture sharing. **Training data, detector layouts, road geometry, calibrated hyperparameters, checkpoints, and experimental scripts are omitted.**

## What is included

The model follows a DeepONet layout with three learnable parts:

1. **Branch CNN** — encodes a historical spatiotemporal window of flow and speed observed at sparse sensors.
2. **Trunk MLP** — evaluates the operator at a query coordinate `(x, t)` after a nonlinear feature expansion.
3. **Parameter CNN** — predicts piecewise traffic-flow parameters (free-flow speed, critical density, capacity) used by the physics-informed part.

The operator output is reconstructed by an inner product of branch and trunk features, then mapped to flow, speed, and density.

## What is intentionally hidden

- Site-specific detector coordinates, ramp locations, and lane maps
- Dataset files and preprocessing pipelines
- Trained weights (`.pat` / `.pth`)
- Normalization statistics computed from real data
- Training losses, learning-rate schedules, and experiment hyperparameters
- Evaluation and visualization scripts

Layer widths and default tensor shapes in this repo are **placeholders**. Replace them with your own configuration before training.

## Quick start

```bash
pip install -r requirements.txt
python example_forward.py
```

`example_forward.py` runs a dummy forward pass with random tensors. It does not load any traffic data.

## File layout

```
models/
  branch_cnn.py      # dual-stream CNN branch
  trunk_mlp.py       # trunk MLP
  param_cnn.py       # traffic-parameter predictor
  pi_deeponet.py     # operator assembly
example_forward.py
```

## Citation

If you use this architecture in academic work, please cite the corresponding PI-DeepONet / GTSP paper.
