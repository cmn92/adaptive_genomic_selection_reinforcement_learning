"""Unit tests for observation discretization."""

import numpy as np

from src.rl.discretizer import ObservationDiscretizer


def main() -> None:
    discretizer = ObservationDiscretizer(
        bins_per_feature=5,
        observation_size=12,
    )

    observation = np.linspace(
        -1.0,
        1.0,
        12,
    )

    state = discretizer.transform(
        observation
    )

    assert len(state) == 12
    assert all(
        0 <= value < 5
        for value in state
    )

    low_state = discretizer.transform(
        np.full(12, -1.0)
    )
    high_state = discretizer.transform(
        np.full(12, 1.0)
    )

    assert low_state == tuple([0] * 12)
    assert high_state == tuple([4] * 12)

    batch = discretizer.transform_batch(
        np.vstack(
            [
                np.full(12, -1.0),
                np.full(12, 1.0),
            ]
        )
    )

    assert len(batch) == 2

    compact = ObservationDiscretizer(
        bins_per_feature=(2, 4, 3),
        observation_size=12,
        feature_indices=(0, 5, 11),
    )

    compact_state = compact.transform(observation)

    assert compact.state_size == 3
    assert compact.feature_indices_ == (0, 5, 11)
    assert compact.number_of_states == 24
    assert len(compact_state) == 3
    assert compact_state == (
        0,
        1,
        2,
    )

    print("All discretizer checks passed.")


if __name__ == "__main__":
    main()
