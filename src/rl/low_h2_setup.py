"""
Shared setup for low-heritability RL gatekeeping experiments.
"""

from __future__ import annotations

from pathlib import Path

from src.environment.breeding_env import (
    BreedingEnv,
    BreedingEnvConfig,
)
from src.environment.r_bridge import RBreedingBridge
from src.environment.reward import RewardConfig
from src.environment.state import observation_size
from src.rl.discretizer import compact_breeding_feature_indices
from src.rl.linear_q import LinearQAgent, LinearQConfig
from src.rl.train import TrainingConfig


DEFAULT_LOW_H2_SEED = 30001
DEFAULT_LOW_H2_HERITABILITY = 0.05
DEFAULT_LOW_H2_POPULATION_SIZE = 500
DEFAULT_LOW_H2_GENERATIONS = 8
DEFAULT_LOW_H2_BUDGET = 75
DEFAULT_LOW_H2_BATCH_SIZE = 25
DEFAULT_LOW_H2_MINIMUM_TRAINING_SIZE = 50
DEFAULT_LOW_H2_PARENTS = 20
DEFAULT_LOW_H2_CROSSES = 50
DEFAULT_LOW_H2_F1_PER_CROSS = 1
DEFAULT_LOW_H2_DH_PER_F1 = 10


def make_low_h2_environment_config(
    *,
    heritability: float = DEFAULT_LOW_H2_HERITABILITY,
    population_size: int = DEFAULT_LOW_H2_POPULATION_SIZE,
    generations: int = DEFAULT_LOW_H2_GENERATIONS,
    maximum_phenotypes: int = DEFAULT_LOW_H2_BUDGET,
    batch_size: int = DEFAULT_LOW_H2_BATCH_SIZE,
    minimum_training_size: int = DEFAULT_LOW_H2_MINIMUM_TRAINING_SIZE,
    number_of_parents: int = DEFAULT_LOW_H2_PARENTS,
    number_of_crosses: int = DEFAULT_LOW_H2_CROSSES,
    f1_per_cross: int = DEFAULT_LOW_H2_F1_PER_CROSS,
    dh_per_f1: int = DEFAULT_LOW_H2_DH_PER_F1,
    reps: int = 1,
    n_cores: int = 1,
    seed: int = DEFAULT_LOW_H2_SEED,
) -> BreedingEnvConfig:
    """Return the focused low-h2 environment used before long RL training."""
    return BreedingEnvConfig(
        maximum_generations=generations,
        batch_size=batch_size,
        minimum_training_size=minimum_training_size,
        maximum_phenotypes=maximum_phenotypes,
        number_of_parents=number_of_parents,
        number_of_crosses=number_of_crosses,
        f1_per_cross=f1_per_cross,
        dh_per_f1=dh_per_f1,
        reps=reps,
        trait=1,
        snp_chip=1,
        n_cores=n_cores,
        seed=seed,
        trait_heritability=heritability,
        population_size=population_size,
    )


def make_gain_reward_config() -> RewardConfig:
    """Return the no-cost, realized-gain RL reward used for fair comparison."""
    return RewardConfig(
        genetic_gain_weight=2.0,
        variance_retention_weight=0.1,
        phenotyping_cost_weight=0.0,
        reliability_improvement_weight=0.0,
        invalid_action_penalty=1.0,
        gain_scale=0.15,
    )


def make_low_h2_env(
    *,
    project_root: str | Path,
    seed: int = DEFAULT_LOW_H2_SEED,
    heritability: float = DEFAULT_LOW_H2_HERITABILITY,
    population_size: int = DEFAULT_LOW_H2_POPULATION_SIZE,
    generations: int = DEFAULT_LOW_H2_GENERATIONS,
    maximum_phenotypes: int = DEFAULT_LOW_H2_BUDGET,
    reps: int = 1,
    n_cores: int = 1,
    reward_config: RewardConfig | None = None,
    population_file: str | Path = "data/initial_candidate_population.RData",
) -> BreedingEnv:
    """Create a low-h2 breeding environment with reset-persistent settings."""
    bridge = RBreedingBridge(
        project_root=project_root,
        population_file=population_file,
        seed=seed,
    )
    config = make_low_h2_environment_config(
        heritability=heritability,
        population_size=population_size,
        generations=generations,
        maximum_phenotypes=maximum_phenotypes,
        reps=reps,
        n_cores=n_cores,
        seed=seed,
    )
    return BreedingEnv(
        bridge=bridge,
        config=config,
        reward_config=reward_config or make_gain_reward_config(),
    )


def make_low_h2_training_config(
    *,
    episodes: int,
    maximum_steps_per_episode: int = 80,
    seed: int = DEFAULT_LOW_H2_SEED,
    checkpoint_every: int = 50,
) -> TrainingConfig:
    """Return training settings for low-h2 RL experiments."""
    return TrainingConfig(
        number_of_episodes=episodes,
        maximum_steps_per_episode=maximum_steps_per_episode,
        seed=seed,
        checkpoint_every=checkpoint_every,
    )


def make_low_h2_linear_agent(
    *,
    number_of_actions: int,
    training_config: TrainingConfig,
    seed: int = DEFAULT_LOW_H2_SEED,
) -> LinearQAgent:
    """Create the compact-feature Linear Q agent for low-h2 training."""
    agent_config = LinearQConfig(
        learning_rate=0.02,
        discount_factor=0.95,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_episodes=max(
            1,
            int(training_config.number_of_episodes * 0.8),
        ),
        l2_penalty=0.001,
        gradient_clip=5.0,
        seed=seed,
    )

    return LinearQAgent(
        number_of_actions=number_of_actions,
        observation_size=observation_size(),
        feature_indices=compact_breeding_feature_indices(),
        config=agent_config,
    )
