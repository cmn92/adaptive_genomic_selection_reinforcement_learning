"""Unit tests for the tabular Q-learning agent."""

from pathlib import Path
import tempfile

import numpy as np

from src.rl.q_learning import (
    QLearningAgent,
    QLearningConfig,
)


def main() -> None:
    agent = QLearningAgent(
        number_of_actions=5,
        config=QLearningConfig(
            learning_rate=0.5,
            discount_factor=0.9,
            epsilon_start=1.0,
            epsilon_end=0.1,
            epsilon_decay_episodes=100,
            seed=12345,
        ),
    )

    state = tuple([0] * 12)
    next_state = tuple([1] * 12)

    action_mask = np.array(
        [True, True, False, False, False]
    )

    action = agent.choose_action(
        state,
        action_mask=action_mask,
        epsilon=1.0,
    )

    assert action in (0, 1)

    td_error = agent.update(
        state=state,
        action=action,
        reward=1.0,
        next_state=next_state,
        next_action_mask=np.array(
            [True, False, False, False, True]
        ),
        terminated=False,
    )

    assert np.isfinite(td_error)
    assert agent.q_values(state)[action] != 0.0

    agent.q_values(state)[1] = 10.0

    greedy = agent.greedy_action(
        state,
        action_mask=np.array(
            [True, True, False, False, False]
        ),
    )

    assert greedy == 1

    with tempfile.TemporaryDirectory() as directory:
        path = (
            Path(directory)
            / "agent.pkl"
        )

        agent.save(path)
        loaded = QLearningAgent.load(path)

        assert loaded.number_of_actions == 5
        assert np.array_equal(
            loaded.q_values(state),
            agent.q_values(state),
        )

    print("All Q-learning checks passed.")


if __name__ == "__main__":
    main()
