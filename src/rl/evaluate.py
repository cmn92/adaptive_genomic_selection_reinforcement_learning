"""
evaluate.py

Evaluate a trained Q-learning policy without exploration.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from src.environment.actions import action_name
from src.environment.breeding_env import BreedingEnv
from src.rl.discretizer import ObservationDiscretizer
from src.rl.q_learning import QLearningAgent


@dataclass
class PolicyEvaluationResult:
    """Outputs from greedy policy evaluation."""

    episode_summary: pd.DataFrame
    step_history: pd.DataFrame
    total_runtime_seconds: float


def evaluate_q_learning(
    *,
    env: BreedingEnv,
    agent: QLearningAgent,
    discretizer: ObservationDiscretizer,
    number_of_episodes: int = 20,
    base_seed: int = 20001,
    maximum_steps_per_episode: int = 200,
) -> PolicyEvaluationResult:
    """Evaluate the greedy learned policy."""
    for name, value in {
        "number_of_episodes": number_of_episodes,
        "base_seed": base_seed,
        "maximum_steps_per_episode": (
            maximum_steps_per_episode
        ),
    }.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            or value < 1
        ):
            raise ValueError(
                f"'{name}' must be a positive integer."
            )

    episode_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []

    evaluation_start = perf_counter()

    for episode in range(1, number_of_episodes + 1):
        episode_seed = base_seed + episode - 1

        observation, info = env.reset(
            seed=episode_seed
        )
        state = discretizer.transform(
            observation
        )

        episode_return = 0.0
        completed_generations = 0
        final_mean_gv = np.nan
        final_gain = np.nan
        final_variance_retention = np.nan

        for step in range(
            1,
            maximum_steps_per_episode + 1,
        ):
            action_mask = np.asarray(
                info["action_mask"],
                dtype=bool,
            )

            action = agent.greedy_action(
                state,
                action_mask=action_mask,
            )

            (
                next_observation,
                reward,
                terminated,
                truncated,
                next_info,
            ) = env.step(action)

            step_rows.append(
                {
                    "episode": episode,
                    "seed": episode_seed,
                    "step": step,
                    "generation": (
                        info["generation"]
                    ),
                    "number_phenotyped_before": (
                        info[
                            "number_phenotyped"
                        ]
                    ),
                    "action": action,
                    "action_name": (
                        action_name(action)
                    ),
                    "reward": float(reward),
                    "event": next_info["event"],
                }
            )

            episode_return += float(reward)

            if (
                next_info["event"]
                == "generation_finalized"
            ):
                completed_generations += 1

                summary = next_info[
                    "cycle_summary"
                ].iloc[0]

                final_mean_gv = float(
                    summary[
                        "next_generation_mean_gv"
                    ]
                )
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

            state = discretizer.transform(
                next_observation
            )
            info = next_info

            if terminated or truncated:
                break

        episode_rows.append(
            {
                "episode": episode,
                "seed": episode_seed,
                "episode_return": (
                    episode_return
                ),
                "steps": step,
                "completed_generations": (
                    completed_generations
                ),
                "final_mean_genetic_value": (
                    final_mean_gv
                ),
                "final_generation_gain": (
                    final_gain
                ),
                "final_variance_retention": (
                    final_variance_retention
                ),
            }
        )

    total_runtime_seconds = (
        perf_counter() - evaluation_start
    )

    return PolicyEvaluationResult(
        episode_summary=pd.DataFrame(
            episode_rows
        ),
        step_history=pd.DataFrame(
            step_rows
        ),
        total_runtime_seconds=(
            total_runtime_seconds
        ),
    )


def save_policy_evaluation(
    result: PolicyEvaluationResult,
    output_directory: str | Path,
) -> dict[str, Path]:
    """Save episode and step evaluation records."""
    if not isinstance(
        result,
        PolicyEvaluationResult,
    ):
        raise TypeError(
            "'result' must be a PolicyEvaluationResult."
        )

    output_directory = Path(
        output_directory
    ).expanduser().resolve()
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    episode_path = (
        output_directory
        / "evaluation_episode_summary.csv"
    )
    steps_path = (
        output_directory
        / "evaluation_step_history.csv"
    )

    result.episode_summary.to_csv(
        episode_path,
        index=False,
    )
    result.step_history.to_csv(
        steps_path,
        index=False,
    )

    return {
        "episode_summary": episode_path,
        "step_history": steps_path,
    }
