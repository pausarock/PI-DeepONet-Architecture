"""Dummy forward pass with random tensors. No real traffic data is used."""

import torch

from models import PIDeepONet


def main() -> None:
    history_len = 8
    num_sensors = 6
    num_segments = 3
    batch = 2

    model = PIDeepONet(
        history_len=history_len,
        num_sensors=num_sensors,
        num_segments=num_segments,
    )
    model.eval()

    u = torch.randn(batch, history_len, num_sensors * 2)
    x = torch.linspace(0.1, 2.9, batch)
    t = torch.zeros(batch)

    with torch.no_grad():
        flow, speed, density = model.operator(u, x, t)
        v_f, k_c, capacity = model.predict_parameters(u, x)

    print("operator output:", flow.shape, speed.shape, density.shape)
    print("parameter output:", v_f.shape, k_c.shape, capacity.shape)


if __name__ == "__main__":
    main()
