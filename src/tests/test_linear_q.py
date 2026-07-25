"""Unit tests for linear Q-learning."""

from tempfile import TemporaryDirectory
from pathlib import Path

import numpy as np

from src.rl.linear_q import LinearQAgent, LinearQConfig
from src.rl.q_learning import load_q_agent


def main() -> None:
    agent = LinearQAgent(
        number_of_actions=3,
        observation_size=4,
        feature_indices=(0, 2),
        config=LinearQConfig(
            learning_rate=0.1,
            discount_factor=0.9,
            epsilon_start=0.0,
            epsilon_end=0.0,
            epsilon_decay_episodes=10,
            seed=12345,
        ),
    )

    state = np.array(
        [0.5, -0.5, 1.0, 0.0],
        dtype=np.float32,
    )
    next_state = np.array(
        [0.25, 0.0, 0.5, -0.5],
        dtype=np.float32,
    )
    action_mask = np.array(
        [True, True, False],
        dtype=bool,
    )

    before = agent.q_values(state)[1]

    td_error = agent.update(
        state=state,
        action=1,
        reward=1.0,
        next_state=next_state,
        next_action_mask=action_mask,
        terminated=False,
    )

    after = agent.q_values(state)[1]

    assert td_error > 0.0
    assert after > before
    assert agent.number_of_features == 2
    assert agent.number_of_parameters == 9
    assert agent.greedy_action(
        state,
        action_mask=action_mask,
    ) == 1

    with TemporaryDirectory() as directory:
        path = Path(directory) / "linear_q.pkl"
        agent.save(path)
        loaded = load_q_agent(path)

        assert isinstance(loaded, LinearQAgent)
        assert np.allclose(
            loaded.weights,
            agent.weights,
        )
        assert loaded.feature_indices == (0, 2)

    print("All linear Q-learning checks passed.")


if __name__ == "__main__":
    main()
