"""Dummy Algorithm-1 loop. No real traffic data is used."""

from models import PIDeepONet
from training import DummyLoaders, TrainConfig, train_loop


def main() -> None:
    history_len = 8
    num_sensors = 6
    num_segments = 3

    model = PIDeepONet(
        history_len=history_len,
        num_sensors=num_sensors,
        num_segments=num_segments,
    )
    loaders = DummyLoaders(history_len, num_sensors, num_segments, batch_size=2)
    config = TrainConfig(
        n_iter=3,
        delta_para=1,
        delta_ramp=2,
        # placeholder ramp zone on the dummy [0, 3] axis
        ramp_zones=((1.2, 1.8),),
    )
    history = train_loop(model, loaders, config)
    print("dummy training finished")
    print({key: [round(v, 4) for v in values] for key, values in history.items()})


if __name__ == "__main__":
    main()
