"""
train.py

Training utilities for the tabular Q-learning breeding agent.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from src.environment.breeding_env import BreedingEnv
from src.rl.discretizer import ObservationDiscretizer
from src.rl.q_learning import QLearningAgent


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for Q-learning training."""

    number_of_episodes: int = 500
    maximum_steps_per_episode: int = 200
    seed: int = 12345
    checkpoint_every: int = 50

    def __post_init__(self) -> None:
        fields = self.__dict__

        for name, value in fields.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or value < 1
            ):
                raise ValueError(
                    f"'{name}' must be a positive integer."
                )


@dataclass
class TrainingResult:
    """Outputs from Q-learning training."""

    episode_history: pd.DataFrame
    agent: QLearningAgent
    discretizer: ObservationDiscretizer
    total_runtime_seconds: float


def train_q_learning(
    *,
    env: BreedingEnv,
    agent: QLearningAgent,
    discretizer: ObservationDiscretizer,
    config: TrainingConfig,
    checkpoint_directory: str | Path | None = None,
) -> TrainingResult:
    """Train a tabular Q-learning agent."""
    if not isinstance(env, BreedingEnv):
        raise TypeError(
            "'env' must be a BreedingEnv instance."
        )

    if not isinstance(agent, QLearningAgent):
        raise TypeError(
            "'agent' must be a QLearningAgent instance."
        )

    if not isinstance(
        discretizer,
        ObservationDiscretizer,
    ):
        raise TypeError(
            "'discretizer' must be an ObservationDiscretizer."
        )

    if not isinstance(config, TrainingConfig):
        raise TypeError(
            "'config' must be a TrainingConfig instance."
        )

    checkpoint_path = None

    if checkpoint_directory is not None:
        checkpoint_path = Path(
            checkpoint_directory
        ).expanduser().resolve()
        checkpoint_path.mkdir(
            parents=True,
            exist_ok=True,
        )

    rows: list[dict[str, Any]] = []
    training_start = perf_counter()

    for episode in range(config.number_of_episodes):
        episode_seed = config.seed + episode

        observation, info = env.reset(
            seed=episode_seed
        )

        state = discretizer.transform(
            observation
        )

        epsilon = agent.epsilon_for_episode(
            episode
        )

        episode_return = 0.0
        episode_steps = 0
        invalid_actions = 0
        finalized_generations = 0
        final_gain = np.nan
        final_variance_retention = np.nan

        for step in range(
            config.maximum_steps_per_episode
        ):
            action_mask = np.asarray(
                info["action_mask"],
                dtype=bool,
            )

            action = agent.choose_action(
                state,
                action_mask=action_mask,
                epsilon=epsilon,
            )

            (
                next_observation,
                reward,
                terminated,
                truncated,
                next_info,
            ) = env.step(action)

            next_state = discretizer.transform(
                next_observation
            )

            agent.update(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                next_action_mask=np.asarray(
                    next_info["action_mask"],
                    dtype=bool,
                ),
                terminated=bool(
                    terminated or truncated
                ),
            )

            episode_return += float(reward)
            episode_steps += 1

            if next_info["event"] == "invalid_action":
                invalid_actions += 1

            if (
                next_info["event"]
                == "generation_finalized"
            ):
                finalized_generations += 1

                summary = next_info[
                    "cycle_summary"
                ].iloc[0]

                final_gain = float(
                    summary[
                        "realized_genetic_gain"
                    ]
                )
                final_variance_retention = float(
                    next_info[
                        "variance_retention"
                    ]
                )

            state = next_state
            info = next_info

            if terminated or truncated:
                break

        rows.append(
            {
                "episode": episode + 1,
                "seed": episode_seed,
                "epsilon": epsilon,
                "episode_return": (
                    episode_return
                ),
                "episode_steps": episode_steps,
                "invalid_actions": (
                    invalid_actions
                ),
                "finalized_generations": (
                    finalized_generations
                ),
                "final_generation_gain": (
                    final_gain
                ),
                "final_variance_retention": (
                    final_variance_retention
                ),
                "q_table_states": len(
                    agent.q_table
                ),
            }
        )

        if (
            checkpoint_path is not None
            and (
                (episode + 1)
                % config.checkpoint_every
                == 0
            )
        ):
            agent.save(
                checkpoint_path
                / f"q_learning_episode_{episode + 1}.pkl"
            )

        print(
            f"Episode {episode + 1}/"
            f"{config.number_of_episodes}; "
            f"return={episode_return:.3f}; "
            f"epsilon={epsilon:.3f}; "
            f"steps={episode_steps}; "
            f"states={len(agent.q_table)}"
        )

    total_runtime_seconds = (
        perf_counter() - training_start
    )

    history = pd.DataFrame(rows)

    return TrainingResult(
        episode_history=history,
        agent=agent,
        discretizer=discretizer,
        total_runtime_seconds=(
            total_runtime_seconds
        ),
    )


def save_training_result(
    result: TrainingResult,
    output_directory: str | Path,
) -> dict[str, Path]:
    """Save training history and final Q-learning agent."""
    if not isinstance(result, TrainingResult):
        raise TypeError(
            "'result' must be a TrainingResult instance."
        )

    output_directory = Path(
        output_directory
    ).expanduser().resolve()
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    history_path = (
        output_directory
        / "training_history.csv"
    )
    agent_path = (
        output_directory
        / "q_learning_agent.pkl"
    )

    result.episode_history.to_csv(
        history_path,
        index=False,
    )
    result.agent.save(agent_path)

    return {
        "training_history": history_path,
        "agent": agent_path,
    }
