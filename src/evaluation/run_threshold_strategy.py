"""
Evaluate a hand-coded diversity-to-PEV threshold policy.

Run from the project root with:

    python -m src.evaluation.run_threshold_strategy --threshold 0.4
"""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from src.environment.actions import PhenotypingAction, action_name
from src.evaluation.metrics import summarize_all_replicates
from src.rl.low_h2_setup import (
    DEFAULT_LOW_H2_GENERATIONS,
    DEFAULT_LOW_H2_HERITABILITY,
    DEFAULT_LOW_H2_POPULATION_SIZE,
    make_gain_reward_config,
    make_low_h2_env,
)


def choose_threshold_action(
    info: dict,
    *,
    reliability_threshold: float,
) -> PhenotypingAction:
    """Choose diversity until reliability is high enough, then PEV."""
    mask = np.asarray(info["action_mask"], dtype=bool)
    mean_reliability = float(
        info.get("mean_reliability", 0.0)
    )

    if (
        mean_reliability >= reliability_threshold
        and mask[int(PhenotypingAction.HIGHEST_PEV)]
    ):
        return PhenotypingAction.HIGHEST_PEV

    if mask[int(PhenotypingAction.DIVERSITY)]:
        return PhenotypingAction.DIVERSITY

    if mask[int(PhenotypingAction.RANDOM)]:
        return PhenotypingAction.RANDOM

    return PhenotypingAction.STOP


def run_threshold_strategy(
    *,
    project_root: str | Path,
    output_directory: str | Path,
    threshold: float,
    number_of_replicates: int,
    base_seed: int,
    heritability: float,
    population_size: int,
    generations: int,
    maximum_phenotypes: int,
    phenotype_reps: int,
    maximum_steps_per_episode: int,
    n_cores: int,
) -> dict[str, Path]:
    """Run matched threshold-rule replicates and save result tables."""
    output_path = Path(output_directory).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    generation_tables = []
    step_rows = []
    start = perf_counter()
    strategy_name = f"threshold_diversity_to_pev_{threshold:g}"

    for replicate in range(1, number_of_replicates + 1):
        seed = base_seed + replicate - 1
        print(
            f"\n=== Threshold replicate {replicate}/"
            f"{number_of_replicates}; seed {seed} ==="
        )
        env = make_low_h2_env(
            project_root=project_root,
            seed=seed,
            heritability=heritability,
            population_size=population_size,
            generations=generations,
            maximum_phenotypes=maximum_phenotypes,
            reps=phenotype_reps,
            n_cores=n_cores,
            reward_config=make_gain_reward_config(),
        )
        _, info = env.reset(seed=seed)
        terminated = truncated = False

        for step in range(1, maximum_steps_per_episode + 1):
            mask = np.asarray(info["action_mask"], dtype=bool)
            action = choose_threshold_action(
                info,
                reliability_threshold=threshold,
            )
            _, reward, terminated, truncated, next_info = env.step(action)

            step_rows.append(
                {
                    "strategy": strategy_name,
                    "replicate": replicate,
                    "run_seed": seed,
                    "step": step,
                    "generation_before": int(info["generation"]),
                    "number_phenotyped_before": int(
                        info["number_phenotyped"]
                    ),
                    "model_available_before": bool(
                        info.get("model_available", False)
                    ),
                    "uncertainty_available_before": bool(
                        info.get("uncertainty_available", False)
                    ),
                    "mean_reliability_before": float(
                        info.get("mean_reliability", 0.0)
                    ),
                    "finite_pev_count_before": int(
                        info.get("finite_pev_count", 0)
                    ),
                    "mask_highest_pev_before": bool(
                        mask[int(PhenotypingAction.HIGHEST_PEV)]
                    ),
                    "action": int(action),
                    "action_name": action_name(action),
                    "reward": float(reward),
                    "event": next_info["event"],
                    "uncertainty_error": info.get(
                        "uncertainty_error",
                        "",
                    ),
                }
            )

            if next_info["event"] == "generation_finalized":
                summary = next_info["cycle_summary"].copy()
                summary.insert(0, "strategy", strategy_name)
                summary.insert(1, "replicate", replicate)
                summary["run_seed"] = seed
                summary["threshold"] = threshold
                summary["episode_step"] = step
                generation_tables.append(summary)
                print(
                    "Generation",
                    int(summary.loc[0, "generation"]),
                    "| gain",
                    round(
                        float(
                            summary.loc[
                                0,
                                "realized_genetic_gain",
                            ]
                        ),
                        3,
                    ),
                )

            info = next_info

            if terminated or truncated:
                break

        if not terminated:
            raise RuntimeError(
                "Threshold policy did not finish within the step limit."
            )

    generation_results = pd.concat(
        generation_tables,
        ignore_index=True,
        sort=False,
    )
    generation_results = generation_results.sort_values(
        ["replicate", "generation"]
    ).reset_index(drop=True)
    replicate_results = summarize_all_replicates(
        generation_results
    )
    step_history = pd.DataFrame(step_rows)

    generation_path = output_path / "generation_results.csv"
    replicate_path = output_path / "replicate_results.csv"
    step_path = output_path / "step_history.csv"
    config_path = output_path / "threshold_configuration.csv"

    generation_results.to_csv(generation_path, index=False)
    replicate_results.to_csv(replicate_path, index=False)
    step_history.to_csv(step_path, index=False)
    pd.DataFrame(
        [
            {
                "threshold": threshold,
                "number_of_replicates": number_of_replicates,
                "base_seed": base_seed,
                "heritability": heritability,
                "population_size": population_size,
                "generations": generations,
                "maximum_phenotypes": maximum_phenotypes,
                "phenotype_reps": phenotype_reps,
                "maximum_steps_per_episode": maximum_steps_per_episode,
                "n_cores": n_cores,
                "total_runtime_seconds": perf_counter() - start,
            }
        ]
    ).to_csv(config_path, index=False)

    print("\nThreshold strategy evaluation complete")
    print(f"Generation results: {generation_path}")
    print(f"Replicate results: {replicate_path}")
    print(f"Step history: {step_path}")

    return {
        "generation_results": generation_path,
        "replicate_results": replicate_path,
        "step_history": step_path,
        "configuration": config_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a diversity-to-PEV threshold strategy."
    )
    parser.add_argument(
        "--output-directory",
        default="results/evaluation/threshold_strategy",
    )
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--replicates", type=int, default=20)
    parser.add_argument("--base-seed", type=int, default=30001)
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
    parser.add_argument("--maximum-phenotypes", type=int, default=75)
    parser.add_argument("--phenotype-reps", type=int, default=2)
    parser.add_argument("--maximum-steps", type=int, default=80)
    parser.add_argument("--n-cores", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]

    run_threshold_strategy(
        project_root=project_root,
        output_directory=args.output_directory,
        threshold=args.threshold,
        number_of_replicates=args.replicates,
        base_seed=args.base_seed,
        heritability=args.heritability,
        population_size=args.population_size,
        generations=args.generations,
        maximum_phenotypes=args.maximum_phenotypes,
        phenotype_reps=args.phenotype_reps,
        maximum_steps_per_episode=args.maximum_steps,
        n_cores=args.n_cores,
    )


if __name__ == "__main__":
    main()
