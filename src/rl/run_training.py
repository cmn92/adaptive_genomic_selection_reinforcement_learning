"""
run_training.py

Executable training script for the tabular Q-learning agent.

Run from the project root with:

    python -m src.rl.run_training

This script:
1. Creates the Python-to-R breeding bridge.
2. Builds the Gymnasium breeding environment.
3. Creates the observation discretizer.
4. Creates the tabular Q-learning agent.
5. Trains the agent.
6. Saves checkpoints, the final agent, and training history.
7. Produces basic training-diagnostic plots.

The default settings are intended for a longer production-style training run.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.environment.breeding_env import (
    BreedingEnv,
    BreedingEnvConfig,
)
from src.environment.r_bridge import RBreedingBridge
from src.environment.reward import RewardConfig
from src.environment.state import observation_size
from src.rl.linear_q import LinearQAgent, LinearQConfig
from src.rl.train import (
    TrainingConfig,
    save_training_result,
    train_q_learning,
)


def rolling_mean(
    values: pd.Series,
    window: int,
) -> pd.Series:
    """Return a rolling mean with sensible behavior near the beginning."""
    return values.rolling(
        window=window,
        min_periods=1,
    ).mean()


def create_training_plots(
    history: pd.DataFrame,
    output_directory: str | Path,
) -> dict[str, Path]:
    """
    Create diagnostic figures from the training history.

    Figures
    -------
    episode_return.png
        Episode reward and its rolling mean.

    epsilon.png
        Exploration rate over training.

    model_size.png
        Number of learned value-function parameters.

    genetic_gain.png
        Final-generation realized gain by episode.

    variance_retention.png
        Final retained genetic variance by episode.
    """
    if not isinstance(history, pd.DataFrame):
        raise TypeError("'history' must be a pandas DataFrame.")

    if history.empty:
        raise ValueError("'history' cannot be empty.")

    output_directory = Path(
        output_directory
    ).expanduser().resolve()

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths: dict[str, Path] = {}

    rolling_window = max(
        5,
        min(25, len(history) // 10 or 1),
    )

    # ------------------------------------------------------------------
    # Episode return
    # ------------------------------------------------------------------

    return_path = output_directory / "episode_return.png"

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(
        history["episode"],
        history["episode_return"],
        alpha=0.4,
        label="Episode return",
    )
    ax.plot(
        history["episode"],
        rolling_mean(
            history["episode_return"],
            rolling_window,
        ),
        label=f"Rolling mean ({rolling_window})",
    )
    ax.set_title("Q-learning episode return")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(return_path, dpi=300)
    plt.close(fig)

    paths["episode_return"] = return_path

    # ------------------------------------------------------------------
    # Epsilon
    # ------------------------------------------------------------------

    epsilon_path = output_directory / "epsilon.png"

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(
        history["episode"],
        history["epsilon"],
    )
    ax.set_title("Exploration rate during training")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Epsilon")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(epsilon_path, dpi=300)
    plt.close(fig)

    paths["epsilon"] = epsilon_path

    # ------------------------------------------------------------------
    # Value-model size
    # ------------------------------------------------------------------

    model_size_path = output_directory / "model_size.png"

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(
        history["episode"],
        history["model_size"],
    )
    ax.set_title("Learned value model size during training")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Model parameters")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(model_size_path, dpi=300)
    plt.close(fig)

    paths["model_size"] = model_size_path

    # ------------------------------------------------------------------
    # Final-generation genetic gain
    # ------------------------------------------------------------------

    gain_path = output_directory / "genetic_gain.png"

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(
        history["episode"],
        history["final_generation_gain"],
        alpha=0.4,
        label="Final-generation gain",
    )
    ax.plot(
        history["episode"],
        rolling_mean(
            history["final_generation_gain"],
            rolling_window,
        ),
        label=f"Rolling mean ({rolling_window})",
    )
    ax.set_title("Realized genetic gain during training")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Realized genetic gain")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(gain_path, dpi=300)
    plt.close(fig)

    paths["genetic_gain"] = gain_path

    # ------------------------------------------------------------------
    # Variance retention
    # ------------------------------------------------------------------

    variance_path = (
        output_directory
        / "variance_retention.png"
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(
        history["episode"],
        history["final_variance_retention"],
        alpha=0.4,
        label="Variance retention",
    )
    ax.plot(
        history["episode"],
        rolling_mean(
            history["final_variance_retention"],
            rolling_window,
        ),
        label=f"Rolling mean ({rolling_window})",
    )
    ax.set_title("Genetic variance retention during training")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Variance retention")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(variance_path, dpi=300)
    plt.close(fig)

    paths["variance_retention"] = variance_path

    return paths


def main() -> None:
    """Run a development Q-learning training experiment."""

    project_root = Path(__file__).resolve().parents[2]
    available_cores = max(
        1,
        min(4, os.cpu_count() or 1),
    )

    output_directory = (
        project_root
        / "results"
        / "rl"
        / "q_learning_development"
    )

    checkpoint_directory = (
        output_directory
        / "checkpoints"
    )

    figures_directory = (
        output_directory
        / "figures"
    )

    # ------------------------------------------------------------------
    # 1. R breeding simulator bridge
    # ------------------------------------------------------------------

    bridge = RBreedingBridge(
        project_root=project_root,
        population_file=(
            "data/initial_candidate_population.RData"
        ),
        seed=12345,
    )

    # ------------------------------------------------------------------
    # 2. RL environment
    # ------------------------------------------------------------------

    environment_config = BreedingEnvConfig(
        maximum_generations=20,
        batch_size=25,
        minimum_training_size=50,
        maximum_phenotypes=200,
        number_of_parents=20,
        number_of_crosses=100,
        f1_per_cross=1,
        dh_per_f1=10,
        reps=1,
        trait=1,
        snp_chip=1,
        n_cores=available_cores,
        seed=12345,
    )

    reward_config = RewardConfig(
        genetic_gain_weight=2.0,
        variance_retention_weight=0.8,
        phenotyping_cost_weight=0.2,
        reliability_improvement_weight=0.3,
        invalid_action_penalty=1.0,
        gain_scale=2.0,
    )

    env = BreedingEnv(
        bridge=bridge,
        config=environment_config,
        reward_config=reward_config,
    )

    # ------------------------------------------------------------------
    # 3. State representation
    #
    # Linear Q uses the continuous observation directly. This avoids the
    # state-recurrence failure mode of tabular Q-learning over many features.
    # ------------------------------------------------------------------

    discretizer = None

    # ------------------------------------------------------------------
    # 4. Linear Q-learning agent
    # ------------------------------------------------------------------

    agent_config = LinearQConfig(
        learning_rate=0.02,
        discount_factor=0.95,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_episodes=250,
        l2_penalty=0.001,
        gradient_clip=5.0,
        seed=12345,
    )

    agent = LinearQAgent(
        number_of_actions=env.action_space.n,
        observation_size=observation_size(),
        config=agent_config,
    )

    # ------------------------------------------------------------------
    # 5. Training settings
    #
    # Function approximation generalizes across continuous states, so hundreds
    # of episodes are a reasonable first run before doing wider sweeps.
    # ------------------------------------------------------------------

    training_config = TrainingConfig(
        number_of_episodes=500,
        maximum_steps_per_episode=220,
        seed=12345,
        checkpoint_every=50,
    )

    # ------------------------------------------------------------------
    # 6. Train
    # ------------------------------------------------------------------

    result = train_q_learning(
        env=env,
        agent=agent,
        discretizer=discretizer,
        config=training_config,
        checkpoint_directory=checkpoint_directory,
    )

    # ------------------------------------------------------------------
    # 7. Save outputs
    # ------------------------------------------------------------------

    saved_paths = save_training_result(
        result,
        output_directory,
    )

    figure_paths = create_training_plots(
        result.episode_history,
        figures_directory,
    )

    configuration_path = (
        output_directory
        / "training_configuration.csv"
    )

    configuration_table = pd.DataFrame(
        [
            {
                **environment_config.__dict__,
                **{
                    f"reward_{key}": value
                    for key, value in reward_config.__dict__.items()
                },
                **{
                    f"agent_{key}": value
                    for key, value in agent_config.__dict__.items()
                },
                **{
                    f"training_{key}": value
                    for key, value in training_config.__dict__.items()
                },
                "agent_kind": agent.kind,
                "agent_observation_size": (
                    agent.observation_size
                ),
                "agent_number_of_features": (
                    agent.number_of_features
                ),
                "agent_number_of_parameters": (
                    agent.number_of_parameters
                ),
                "total_runtime_seconds": (
                    result.total_runtime_seconds
                ),
                "final_model_size": (
                    result.episode_history[
                        "model_size"
                    ].iloc[-1]
                ),
            }
        ]
    )

    configuration_table.to_csv(
        configuration_path,
        index=False,
    )

    print("\nLinear Q-learning training completed.")
    print(
        "Total runtime seconds:",
        round(
            result.total_runtime_seconds,
            3,
        ),
    )
    print("Model parameters:", agent.number_of_parameters)

    print("\nSaved training files:")
    for name, path in saved_paths.items():
        print(f"{name}: {path}")

    print(
        f"configuration: {configuration_path}"
    )

    print("\nSaved figures:")
    for name, path in figure_paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
