"""
Short Linear Q smoke test before expensive low-h2 training.

Run from the project root with:

    python -m src.rl.run_smoke_training --episodes 30
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.environment.actions import PhenotypingAction
from src.rl.evaluate import (
    evaluate_q_learning,
    save_policy_evaluation,
)
from src.rl.low_h2_setup import (
    DEFAULT_LOW_H2_GENERATIONS,
    DEFAULT_LOW_H2_HERITABILITY,
    DEFAULT_LOW_H2_POPULATION_SIZE,
    DEFAULT_LOW_H2_SEED,
    make_gain_reward_config,
    make_low_h2_env,
    make_low_h2_linear_agent,
    make_low_h2_training_config,
)
from src.rl.train import save_training_result, train_q_learning


def run_smoke_training(
    *,
    project_root: str | Path,
    output_directory: str | Path,
    episodes: int,
    eval_episodes: int,
    seed: int,
    heritability: float,
    population_size: int,
    generations: int,
    maximum_steps_per_episode: int,
    n_cores: int,
) -> dict[str, Path]:
    """Train a tiny low-h2 Linear Q run and evaluate its greedy policy."""
    output_path = Path(output_directory).expanduser().resolve()
    training_path = output_path / "training"
    evaluation_path = output_path / "greedy_evaluation"
    checkpoint_path = training_path / "checkpoints"
    output_path.mkdir(parents=True, exist_ok=True)

    reward_config = make_gain_reward_config()
    env = make_low_h2_env(
        project_root=project_root,
        seed=seed,
        heritability=heritability,
        population_size=population_size,
        generations=generations,
        maximum_phenotypes=75,
        n_cores=n_cores,
        reward_config=reward_config,
    )
    training_config = make_low_h2_training_config(
        episodes=episodes,
        maximum_steps_per_episode=maximum_steps_per_episode,
        seed=seed,
        checkpoint_every=max(1, min(50, episodes)),
    )
    agent = make_low_h2_linear_agent(
        number_of_actions=env.action_space.n,
        training_config=training_config,
        seed=seed,
    )

    training_result = train_q_learning(
        env=env,
        agent=agent,
        discretizer=None,
        config=training_config,
        checkpoint_directory=checkpoint_path,
    )
    training_outputs = save_training_result(
        training_result,
        training_path,
    )

    evaluation_result = evaluate_q_learning(
        env=env,
        agent=training_result.agent,
        discretizer=None,
        number_of_episodes=eval_episodes,
        base_seed=seed + 50000,
        maximum_steps_per_episode=maximum_steps_per_episode,
    )
    evaluation_outputs = save_policy_evaluation(
        evaluation_result,
        evaluation_path,
    )

    history = training_result.episode_history
    step_history = evaluation_result.step_history
    train_pev_available = int(
        history["pev_available_steps"].sum()
    )
    train_pev_chosen = int(
        history["action_highest_pev_count"].sum()
    )
    greedy_pev_available = int(
        step_history["mask_highest_pev_before"].sum()
    )
    greedy_pev_chosen = int(
        (
            step_history["action"]
            == int(PhenotypingAction.HIGHEST_PEV)
        ).sum()
    )

    summary = pd.DataFrame(
        [
            {
                "episodes": episodes,
                "eval_episodes": eval_episodes,
                "heritability": heritability,
                "population_size": population_size,
                "generations": generations,
                "train_pev_available_steps": train_pev_available,
                "train_highest_pev_actions": train_pev_chosen,
                "greedy_pev_available_steps": greedy_pev_available,
                "greedy_highest_pev_actions": greedy_pev_chosen,
                "final_rolling_return": float(
                    history["episode_return"].tail(5).mean()
                ),
            }
        ]
    )
    summary_path = output_path / "smoke_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\nLinear Q smoke test complete")
    print(f"Training history: {training_outputs['training_history']}")
    print(f"Greedy steps: {evaluation_outputs['step_history']}")
    print(f"Summary: {summary_path}")
    print(f"Training PEV available steps: {train_pev_available}")
    print(f"Training Highest-PEV actions: {train_pev_chosen}")
    print(f"Greedy PEV available steps: {greedy_pev_available}")
    print(f"Greedy Highest-PEV actions: {greedy_pev_chosen}")

    return {
        **training_outputs,
        **evaluation_outputs,
        "summary": summary_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a short low-h2 Linear Q smoke test."
    )
    parser.add_argument(
        "--output-directory",
        default="results/rl/linear_q_smoke",
    )
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=DEFAULT_LOW_H2_SEED)
    parser.add_argument(
        "--heritability",
        type=float,
        default=DEFAULT_LOW_H2_HERITABILITY,
    )
    parser.add_argument(
        "--population-size",
        type=int,
        default=DEFAULT_LOW_H2_POPULATION_SIZE,
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=DEFAULT_LOW_H2_GENERATIONS,
    )
    parser.add_argument("--maximum-steps", type=int, default=80)
    parser.add_argument("--n-cores", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]

    run_smoke_training(
        project_root=project_root,
        output_directory=args.output_directory,
        episodes=args.episodes,
        eval_episodes=args.eval_episodes,
        seed=args.seed,
        heritability=args.heritability,
        population_size=args.population_size,
        generations=args.generations,
        maximum_steps_per_episode=args.maximum_steps,
        n_cores=args.n_cores,
    )


if __name__ == "__main__":
    main()
