"""
q_learning.py

Tabular Q-learning agent with support for invalid-action masks.

The Q-table is stored sparsely in a Python dictionary. Only states actually
visited during training consume memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Any, Hashable

import numpy as np


DiscreteState = tuple[int, ...]


@dataclass(frozen=True)
class QLearningConfig:
    """Hyperparameters for tabular Q-learning."""

    learning_rate: float = 0.1
    discount_factor: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_episodes: int = 500
    seed: int = 12345

    def __post_init__(self) -> None:
        numeric_fields = {
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor,
            "epsilon_start": self.epsilon_start,
            "epsilon_end": self.epsilon_end,
        }

        for name, value in numeric_fields.items():
            numeric = float(value)

            if not np.isfinite(numeric):
                raise ValueError(
                    f"'{name}' must be finite."
                )

            if not 0.0 <= numeric <= 1.0:
                raise ValueError(
                    f"'{name}' must lie between zero and one."
                )

        if self.epsilon_end > self.epsilon_start:
            raise ValueError(
                "'epsilon_end' cannot exceed 'epsilon_start'."
            )

        if (
            isinstance(self.epsilon_decay_episodes, bool)
            or not isinstance(
                self.epsilon_decay_episodes,
                (int, np.integer),
            )
            or self.epsilon_decay_episodes < 1
        ):
            raise ValueError(
                "'epsilon_decay_episodes' must be a positive integer."
            )

        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, (int, np.integer))
        ):
            raise TypeError("'seed' must be an integer.")


class QLearningAgent:
    """
    Sparse tabular Q-learning agent.

    Parameters
    ----------
    number_of_actions:
        Size of the environment's discrete action space.

    config:
        Q-learning hyperparameters.
    """

    def __init__(
        self,
        *,
        number_of_actions: int,
        config: QLearningConfig | None = None,
    ) -> None:
        if (
            isinstance(number_of_actions, bool)
            or not isinstance(
                number_of_actions,
                (int, np.integer),
            )
            or number_of_actions < 1
        ):
            raise ValueError(
                "'number_of_actions' must be a positive integer."
            )

        self.number_of_actions = int(
            number_of_actions
        )
        self.config = config or QLearningConfig()

        self._rng = np.random.default_rng(
            self.config.seed
        )

        self.q_table: dict[
            DiscreteState,
            np.ndarray,
        ] = {}

    def epsilon_for_episode(
        self,
        episode: int,
    ) -> float:
        """Return linearly decayed epsilon for one episode."""
        if (
            isinstance(episode, bool)
            or not isinstance(episode, (int, np.integer))
            or episode < 0
        ):
            raise ValueError(
                "'episode' must be a nonnegative integer."
            )

        fraction = min(
            episode / self.config.epsilon_decay_episodes,
            1.0,
        )

        epsilon = (
            self.config.epsilon_start
            + fraction
            * (
                self.config.epsilon_end
                - self.config.epsilon_start
            )
        )

        return float(epsilon)

    def q_values(
        self,
        state: DiscreteState,
    ) -> np.ndarray:
        """Return the Q-values for a state, creating them if necessary."""
        if state not in self.q_table:
            self.q_table[state] = np.zeros(
                self.number_of_actions,
                dtype=np.float64,
            )

        return self.q_table[state]

    def choose_action(
        self,
        state: DiscreteState,
        *,
        action_mask: np.ndarray,
        epsilon: float,
    ) -> int:
        """Choose an epsilon-greedy action from valid actions only."""
        mask = np.asarray(
            action_mask,
            dtype=bool,
        )

        if mask.shape != (self.number_of_actions,):
            raise ValueError(
                "'action_mask' has an unexpected shape."
            )

        valid_actions = np.flatnonzero(mask)

        if valid_actions.size == 0:
            raise ValueError(
                "At least one valid action is required."
            )

        epsilon = float(epsilon)

        if not 0.0 <= epsilon <= 1.0:
            raise ValueError(
                "'epsilon' must lie between zero and one."
            )

        if self._rng.random() < epsilon:
            return int(
                self._rng.choice(valid_actions)
            )

        values = self.q_values(state)
        valid_values = values[valid_actions]
        best_value = np.max(valid_values)

        best_actions = valid_actions[
            np.isclose(
                valid_values,
                best_value,
            )
        ]

        return int(
            self._rng.choice(best_actions)
        )

    def update(
        self,
        *,
        state: DiscreteState,
        action: int,
        reward: float,
        next_state: DiscreteState,
        next_action_mask: np.ndarray,
        terminated: bool,
    ) -> float:
        """
        Apply the standard one-step Q-learning update.

        Returns
        -------
        float
            The temporal-difference error.
        """
        if (
            isinstance(action, bool)
            or not isinstance(action, (int, np.integer))
            or not 0 <= int(action) < self.number_of_actions
        ):
            raise ValueError("'action' is outside the action space.")

        reward = float(reward)

        if not np.isfinite(reward):
            raise ValueError("'reward' must be finite.")

        current_values = self.q_values(state)
        current_q = float(current_values[int(action)])

        if terminated:
            target = reward
        else:
            mask = np.asarray(
                next_action_mask,
                dtype=bool,
            )

            if mask.shape != (self.number_of_actions,):
                raise ValueError(
                    "'next_action_mask' has an unexpected shape."
                )

            valid_next_actions = np.flatnonzero(mask)

            if valid_next_actions.size == 0:
                target = reward
            else:
                next_values = self.q_values(next_state)
                target = (
                    reward
                    + self.config.discount_factor
                    * np.max(
                        next_values[valid_next_actions]
                    )
                )

        td_error = target - current_q

        current_values[int(action)] = (
            current_q
            + self.config.learning_rate
            * td_error
        )

        return float(td_error)

    def greedy_action(
        self,
        state: DiscreteState,
        *,
        action_mask: np.ndarray,
    ) -> int:
        """Choose a greedy valid action."""
        return self.choose_action(
            state,
            action_mask=action_mask,
            epsilon=0.0,
        )

    def save(
        self,
        path: str | Path,
    ) -> Path:
        """Save the agent configuration and Q-table."""
        path = Path(path).expanduser().resolve()
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = {
            "agent_type": "tabular_q",
            "number_of_actions": self.number_of_actions,
            "config": self.config,
            "q_table": self.q_table,
        }

        with path.open("wb") as file:
            pickle.dump(payload, file)

        return path

    @classmethod
    def load(
        cls,
        path: str | Path,
    ) -> "QLearningAgent":
        """Load a previously saved agent."""
        path = Path(path).expanduser().resolve()

        if not path.is_file():
            raise FileNotFoundError(
                f"Saved agent was not found: {path}"
            )

        with path.open("rb") as file:
            payload = pickle.load(file)

        agent = cls(
            number_of_actions=payload[
                "number_of_actions"
            ],
            config=payload["config"],
        )
        agent.q_table = payload["q_table"]

        return agent


def load_q_agent(
    path: str | Path,
) -> QLearningAgent | Any:
    """Load either a tabular or linear Q-learning agent."""
    path = Path(path).expanduser().resolve()

    if not path.is_file():
        raise FileNotFoundError(
            f"Saved agent was not found: {path}"
        )

    with path.open("rb") as file:
        payload = pickle.load(file)

    agent_type = payload.get("agent_type", "tabular_q")

    if agent_type == "tabular_q":
        agent = QLearningAgent(
            number_of_actions=payload[
                "number_of_actions"
            ],
            config=payload["config"],
        )
        agent.q_table = payload["q_table"]
        return agent

    if agent_type == "linear_q":
        from src.rl.linear_q import LinearQAgent

        return LinearQAgent.load(path)

    raise ValueError(
        f"Unknown saved Q-agent type: {agent_type!r}."
    )
