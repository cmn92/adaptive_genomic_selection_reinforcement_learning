"""
linear_q.py

Linear Q-learning agent for continuous breeding-environment observations.

Unlike tabular Q-learning, this agent learns one linear value function per
action and therefore generalizes across nearby continuous states.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle

import numpy as np


@dataclass(frozen=True)
class LinearQConfig:
    """Hyperparameters for semi-gradient linear Q-learning."""

    learning_rate: float = 0.02
    discount_factor: float = 0.95
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_episodes: int = 250
    l2_penalty: float = 0.0
    gradient_clip: float = 5.0
    seed: int = 12345

    def __post_init__(self) -> None:
        bounded_fields = {
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor,
            "epsilon_start": self.epsilon_start,
            "epsilon_end": self.epsilon_end,
        }

        for name, value in bounded_fields.items():
            numeric = float(value)

            if not np.isfinite(numeric):
                raise ValueError(f"'{name}' must be finite.")
            if not 0.0 <= numeric <= 1.0:
                raise ValueError(
                    f"'{name}' must lie between zero and one."
                )

        if self.epsilon_end > self.epsilon_start:
            raise ValueError(
                "'epsilon_end' cannot exceed 'epsilon_start'."
            )

        for name, value in {
            "epsilon_decay_episodes": self.epsilon_decay_episodes,
            "seed": self.seed,
        }.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or value < 1
            ):
                raise ValueError(
                    f"'{name}' must be a positive integer."
                )

        for name, value in {
            "l2_penalty": self.l2_penalty,
            "gradient_clip": self.gradient_clip,
        }.items():
            numeric = float(value)

            if not np.isfinite(numeric):
                raise ValueError(f"'{name}' must be finite.")
            if numeric < 0.0:
                raise ValueError(f"'{name}' cannot be negative.")


class LinearQAgent:
    """
    Linear Q-learning agent with one weight vector per action.

    Parameters
    ----------
    number_of_actions:
        Size of the environment's discrete action space.

    observation_size:
        Width of the continuous observation vector.

    feature_indices:
        Optional subset of observation columns to use. If omitted, all
        observation features are used.

    config:
        Linear Q-learning hyperparameters.
    """

    kind = "linear_q"

    def __init__(
        self,
        *,
        number_of_actions: int,
        observation_size: int,
        feature_indices: tuple[int, ...] | None = None,
        config: LinearQConfig | None = None,
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

        if (
            isinstance(observation_size, bool)
            or not isinstance(observation_size, (int, np.integer))
            or observation_size < 1
        ):
            raise ValueError(
                "'observation_size' must be a positive integer."
            )

        self.number_of_actions = int(number_of_actions)
        self.observation_size = int(observation_size)
        self.feature_indices = self._normalize_feature_indices(
            feature_indices
        )
        self.config = config or LinearQConfig()

        self._rng = np.random.default_rng(
            self.config.seed
        )

        self.weights = np.zeros(
            (
                self.number_of_actions,
                self.number_of_features + 1,
            ),
            dtype=np.float64,
        )

    def _normalize_feature_indices(
        self,
        feature_indices: tuple[int, ...] | None,
    ) -> tuple[int, ...]:
        """Validate and standardize selected observation columns."""
        if feature_indices is None:
            return tuple(range(self.observation_size))

        if not isinstance(feature_indices, tuple):
            raise TypeError(
                "'feature_indices' must be a tuple or None."
            )

        if len(feature_indices) < 1:
            raise ValueError(
                "'feature_indices' must contain at least one index."
            )

        normalized: list[int] = []

        for position, value in enumerate(feature_indices):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
            ):
                raise TypeError(
                    f"Feature index at position {position} must be an integer."
                )

            index = int(value)

            if not 0 <= index < self.observation_size:
                raise ValueError(
                    f"Feature index {index} is outside the observation width."
                )

            normalized.append(index)

        if len(set(normalized)) != len(normalized):
            raise ValueError(
                "'feature_indices' cannot contain duplicates."
            )

        return tuple(normalized)

    @property
    def number_of_features(self) -> int:
        """Return the number of continuous features used by the model."""
        return len(self.feature_indices)

    @property
    def number_of_parameters(self) -> int:
        """Return the total number of learned scalar parameters."""
        return int(self.weights.size)

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

    def _features(
        self,
        observation: np.ndarray,
    ) -> np.ndarray:
        """Return clipped selected features with an intercept prepended."""
        values = np.asarray(
            observation,
            dtype=np.float64,
        )

        if values.shape != (self.observation_size,):
            raise ValueError(
                "'observation' must have shape "
                f"({self.observation_size},)."
            )

        if not np.isfinite(values).all():
            raise ValueError(
                "'observation' cannot contain NaN or infinity."
            )

        selected = np.clip(
            values[list(self.feature_indices)],
            -1.0,
            1.0,
        )

        return np.concatenate(
            [
                np.array([1.0], dtype=np.float64),
                selected,
            ]
        )

    def q_values(
        self,
        state: np.ndarray,
    ) -> np.ndarray:
        """Return Q-values for every action."""
        features = self._features(state)
        return self.weights @ features

    def choose_action(
        self,
        state: np.ndarray,
        *,
        action_mask: np.ndarray,
        epsilon: float,
    ) -> int:
        """Choose an epsilon-greedy valid action."""
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
            np.isclose(valid_values, best_value)
        ]

        return int(
            self._rng.choice(best_actions)
        )

    def update(
        self,
        *,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        next_action_mask: np.ndarray,
        terminated: bool,
    ) -> float:
        """
        Apply one semi-gradient Q-learning update.

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

        features = self._features(state)
        action = int(action)
        current_q = float(
            self.weights[action] @ features
        )

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
                    * float(
                        np.max(
                            next_values[valid_next_actions]
                        )
                    )
                )

        td_error = target - current_q

        if self.config.gradient_clip > 0.0:
            td_error = float(
                np.clip(
                    td_error,
                    -self.config.gradient_clip,
                    self.config.gradient_clip,
                )
            )

        regularization = (
            self.config.l2_penalty
            * self.weights[action]
        )

        self.weights[action] += (
            self.config.learning_rate
            * (
                td_error * features
                - regularization
            )
        )

        return float(td_error)

    def greedy_action(
        self,
        state: np.ndarray,
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
        """Save the agent configuration and learned weights."""
        path = Path(path).expanduser().resolve()
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = {
            "agent_type": self.kind,
            "number_of_actions": self.number_of_actions,
            "observation_size": self.observation_size,
            "feature_indices": self.feature_indices,
            "config": self.config,
            "weights": self.weights,
        }

        with path.open("wb") as file:
            pickle.dump(payload, file)

        return path

    @classmethod
    def load(
        cls,
        path: str | Path,
    ) -> "LinearQAgent":
        """Load a previously saved linear Q agent."""
        path = Path(path).expanduser().resolve()

        if not path.is_file():
            raise FileNotFoundError(
                f"Saved agent was not found: {path}"
            )

        with path.open("rb") as file:
            payload = pickle.load(file)

        if payload.get("agent_type") != cls.kind:
            raise ValueError(
                "Saved payload is not a linear Q agent."
            )

        agent = cls(
            number_of_actions=payload["number_of_actions"],
            observation_size=payload["observation_size"],
            feature_indices=payload["feature_indices"],
            config=payload["config"],
        )
        agent.weights = np.asarray(
            payload["weights"],
            dtype=np.float64,
        )

        if agent.weights.shape != (
            agent.number_of_actions,
            agent.number_of_features + 1,
        ):
            raise ValueError(
                "Saved linear Q weights have an unexpected shape."
            )

        return agent
