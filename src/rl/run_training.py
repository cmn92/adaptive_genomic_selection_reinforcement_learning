"""
run_training.py

Executable training script for the linear Q-learning agent.

Run from the project root with:

    python -m src.rl.run_training

This script:
1. Creates the Python-to-R breeding bridge.
2. Builds the Gymnasium breeding environment.
3. Uses compact continuous state features.
4. Creates the linear Q-learning agent.
5. Trains the agent.
6. Saves checkpoints, the final agent, and training history.
7. Produces basic training-diagnostic plots.

The default settings are intended for a longer production-style training run.
"""

from __future__ import annotations

import argparse
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
from src.rl.discretizer import compact_breeding_feature_indices
from src.rl.linear_q import LinearQAgent, LinearQConfig
from src.rl.train import (
    TrainingConfig,
    save_training_result,
    train_q_learning,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the low-h2 Linear Q run."""
    parser = argparse.ArgumentParser(
        description="Train the compact-feature Linear Q agent."
    )
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=30001)
    parser.add_argument("--heritability", type=float, default=0.05)
    parser.add_argument("--population-size", type=int, default=500)
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--maximum-phenotypes", type=int, default=75)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--minimum-training-size", type=int, default=50)
    parser.add_argument("--number-of-parents", type=int, default=20)
    parser.add_argument("--number-of-crosses", type=int, default=50)
    parser.add_argument("--dh-per-f1", type=int, default=10)
    parser.add_argument("--maximum-steps", type=int, default=80)
    parser.add_argument("--n-cores", type=int, default=None)
    parser.add_argument("--output-directory", default=None)
    return parser.parse_args()


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

    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    available_cores = max(
        1,
        min(4, os.cpu_count() or 1),
    )
    n_cores = (
        available_cores
        if args.n_cores is None
        else max(1, int(args.n_cores))
    )

    if args.output_directory is None:
        heritability_label = str(args.heritability).replace(
            ".",
            "_",
        )
        output_directory = (
            project_root
            / "results"
            / "rl"
            / f"linear_q_h2_{heritability_label}"
        )
    else:
        output_directory = Path(args.output_directory)
        if not output_directory.is_absolute():
            output_directory = project_root / output_directory

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
        seed=args.seed,
    )

    # ------------------------------------------------------------------
    # 2. RL environment
    # ------------------------------------------------------------------

    environment_config = BreedingEnvConfig(
        maximum_generations=args.generations,
        batch_size=args.batch_size,
        minimum_training_size=args.minimum_training_size,
        maximum_phenotypes=args.maximum_phenotypes,
        number_of_parents=args.number_of_parents,
        number_of_crosses=args.number_of_crosses,
        f1_per_cross=1,
        dh_per_f1=args.dh_per_f1,
        reps=1,
        trait=1,
        snp_chip=1,
        n_cores=n_cores,
        seed=args.seed,
        trait_heritability=args.heritability,
        population_size=args.population_size,
    )

    reward_config = RewardConfig(
        genetic_gain_weight=2.0,
        variance_retention_weight=0.8,
        # Fixed-budget baselines always spend the full phenotyping budget.
        # Keep Linear Q focused on batch choice rather than budget avoidance.
        phenotyping_cost_weight=0.0,
        reliability_improvement_weight=0.0,
        invalid_action_penalty=1.0,
        gain_scale=5.0,
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
    # 4. Training settings
    #
    # Function approximation generalizes across continuous states, so hundreds
    # of episodes are a reasonable first run after the diagnostic and smoke
    # test prove that PEV is available and used.
    # ------------------------------------------------------------------

    training_config = TrainingConfig(
        number_of_episodes=args.episodes,
        maximum_steps_per_episode=args.maximum_steps,
        seed=args.seed,
        checkpoint_every=50,
    )

    # ------------------------------------------------------------------
    # 5. Linear Q-learning agent
    # ------------------------------------------------------------------

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
        seed=args.seed,
    )

    agent = LinearQAgent(
        number_of_actions=env.action_space.n,
        observation_size=observation_size(),
        feature_indices=compact_breeding_feature_indices(),
        config=agent_config,
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
                "agent_feature_indices": (
                    ",".join(
                        str(index)
                        for index in agent.feature_indices
                    )
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
