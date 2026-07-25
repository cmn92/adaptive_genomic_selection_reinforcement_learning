from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any
import numpy as np
import pandas as pd

from src.environment.breeding_env import BreedingEnv, BreedingEnvConfig
from src.environment.r_bridge import RBreedingBridge
from src.environment.reward import RewardConfig
from src.evaluation.metrics import summarize_all_replicates
from src.rl.discretizer import ObservationDiscretizer
from src.rl.q_learning import QLearningAgent


@dataclass(frozen=True)
class RLStrategyEvaluationConfig:
    number_of_replicates: int = 20
    number_of_generations: int = 20
    batch_size: int = 25
    minimum_training_size: int = 50
    maximum_phenotypes: int = 200
    number_of_parents: int = 20
    number_of_crosses: int = 100
    f1_per_cross: int = 1
    dh_per_f1: int = 10
    reps: int = 1
    trait: int = 1
    snp_chip: int = 1
    n_cores: int = 1
    base_seed: int = 20001
    maximum_steps_per_episode: int = 200
    strategy_name: str = "q_learning"


@dataclass
class RLStrategyEvaluationResult:
    config: RLStrategyEvaluationConfig
    generation_results: pd.DataFrame
    replicate_results: pd.DataFrame
    step_history: pd.DataFrame
    total_runtime_seconds: float


def evaluate_frozen_rl_strategy(
    *,
    project_root: str | Path,
    agent: QLearningAgent,
    discretizer: ObservationDiscretizer,
    config: RLStrategyEvaluationConfig,
    population_file: str | Path = "data/initial_candidate_population.RData",
    reward_config: RewardConfig | None = None,
) -> RLStrategyEvaluationResult:
    project_root = Path(project_root).expanduser().resolve()
    reward_config = reward_config or RewardConfig()

    if agent.q_table:
        state_widths = {
            len(state)
            for state in agent.q_table
        }
        expected_width = discretizer.observation_size

        if state_widths != {expected_width}:
            raise ValueError(
                "The loaded agent was trained with incompatible "
                "discrete-state widths. Retrain the Q-learning agent "
                f"for observation size {expected_width} before "
                "running frozen-policy evaluation."
            )

    generation_tables = []
    step_rows: list[dict[str, Any]] = []
    start = perf_counter()

    for replicate in range(1, config.number_of_replicates + 1):
        seed = config.base_seed + replicate - 1
        print(f"\n=== RL replicate {replicate}/{config.number_of_replicates}; seed {seed} ===")

        bridge = RBreedingBridge(
            project_root=project_root,
            population_file=population_file,
            seed=seed,
        )
        env = BreedingEnv(
            bridge=bridge,
            config=BreedingEnvConfig(
                maximum_generations=config.number_of_generations,
                batch_size=config.batch_size,
                minimum_training_size=config.minimum_training_size,
                maximum_phenotypes=config.maximum_phenotypes,
                number_of_parents=config.number_of_parents,
                number_of_crosses=config.number_of_crosses,
                f1_per_cross=config.f1_per_cross,
                dh_per_f1=config.dh_per_f1,
                reps=config.reps,
                trait=config.trait,
                snp_chip=config.snp_chip,
                n_cores=config.n_cores,
                seed=seed,
            ),
            reward_config=reward_config,
        )

        observation, info = env.reset(seed=seed)
        terminated = truncated = False

        for step_number in range(1, config.maximum_steps_per_episode + 1):
            state = discretizer.transform(observation)
            action = agent.greedy_action(
                state,
                action_mask=np.asarray(info["action_mask"], dtype=bool),
            )
            next_observation, reward, terminated, truncated, next_info = env.step(action)

            step_rows.append({
                "strategy": config.strategy_name,
                "replicate": replicate,
                "run_seed": seed,
                "step": step_number,
                "generation_before": info["generation"],
                "number_phenotyped_before": info["number_phenotyped"],
                "action": int(action),
                "action_name": next_info.get("action_name", str(action)),
                "reward": float(reward),
                "event": next_info["event"],
            })

            if next_info["event"] == "generation_finalized":
                summary = next_info["cycle_summary"].copy()
                summary.insert(0, "strategy", config.strategy_name)
                summary.insert(1, "replicate", replicate)
                summary["run_seed"] = seed
                summary["cycle_seed"] = seed + int(summary.loc[0, "generation"]) - 1
                summary["episode_step"] = step_number
                generation_tables.append(summary)
                print(
                    "Generation",
                    int(summary.loc[0, "generation"]),
                    "| phenotyped",
                    int(summary.loc[0, "number_phenotyped"]),
                    "| gain",
                    round(float(summary.loc[0, "realized_genetic_gain"]), 3),
                )

            observation, info = next_observation, next_info
            if terminated or truncated:
                break

        if not terminated:
            raise RuntimeError("Frozen RL policy did not finish within the step limit.")

    generation_results = pd.concat(generation_tables, ignore_index=True, sort=False)
    generation_results = generation_results.sort_values(
        ["replicate", "generation"]
    ).reset_index(drop=True)

    return RLStrategyEvaluationResult(
        config=config,
        generation_results=generation_results,
        replicate_results=summarize_all_replicates(generation_results),
        step_history=pd.DataFrame(step_rows),
        total_runtime_seconds=perf_counter() - start,
    )
