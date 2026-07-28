"""
Short rollout diagnostic for Highest-PEV action availability.

Run from the project root with:

    python -m src.rl.run_pev_diagnostic
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.environment.actions import PhenotypingAction, action_name
from src.rl.low_h2_setup import (
    DEFAULT_LOW_H2_GENERATIONS,
    DEFAULT_LOW_H2_HERITABILITY,
    DEFAULT_LOW_H2_POPULATION_SIZE,
    DEFAULT_LOW_H2_SEED,
    make_low_h2_env,
)


def choose_diagnostic_action(
    info: dict,
) -> PhenotypingAction:
    """Choose PEV whenever it is available after model fitting."""
    mask = np.asarray(info["action_mask"], dtype=bool)

    if mask[int(PhenotypingAction.HIGHEST_PEV)]:
        return PhenotypingAction.HIGHEST_PEV

    if mask[int(PhenotypingAction.DIVERSITY)]:
        return PhenotypingAction.DIVERSITY

    if mask[int(PhenotypingAction.RANDOM)]:
        return PhenotypingAction.RANDOM

    return PhenotypingAction.STOP


def info_row(
    *,
    step: int,
    seed: int,
    info: dict,
    action: PhenotypingAction,
    reward: float | None = None,
    next_info: dict | None = None,
) -> dict:
    """Return one diagnostic row for the decision point."""
    mask = np.asarray(info["action_mask"], dtype=bool)
    row = {
        "seed": seed,
        "step": step,
        "generation_before": int(info["generation"]),
        "number_phenotyped_before": int(info["number_phenotyped"]),
        "model_available_before": bool(
            info.get("model_available", False)
        ),
        "uncertainty_available_before": bool(
            info.get("uncertainty_available", False)
        ),
        "finite_pev_count_before": int(
            info.get("finite_pev_count", 0)
        ),
        "mean_pev_before": float(info.get("mean_pev", 0.0)),
        "max_pev_before": float(info.get("max_pev", 0.0)),
        "mean_reliability_before": float(
            info.get("mean_reliability", 0.0)
        ),
        "mask_random_before": bool(
            mask[int(PhenotypingAction.RANDOM)]
        ),
        "mask_diversity_before": bool(
            mask[int(PhenotypingAction.DIVERSITY)]
        ),
        "mask_highest_pev_before": bool(
            mask[int(PhenotypingAction.HIGHEST_PEV)]
        ),
        "mask_highest_gebv_before": bool(
            mask[int(PhenotypingAction.HIGHEST_GEBV)]
        ),
        "mask_stop_before": bool(mask[int(PhenotypingAction.STOP)]),
        "chosen_action": int(action),
        "chosen_action_name": action_name(action),
        "uncertainty_error_before": info.get(
            "uncertainty_error",
            "",
        ),
    }

    if reward is not None:
        row["reward"] = float(reward)

    if next_info is not None:
        row["event_after"] = next_info["event"]
        row["generation_after"] = int(next_info["generation"])
        row["number_phenotyped_after"] = int(
            next_info["number_phenotyped"]
        )
        row["model_available_after"] = bool(
            next_info.get("model_available", False)
        )
        row["uncertainty_available_after"] = bool(
            next_info.get("uncertainty_available", False)
        )
        row["finite_pev_count_after"] = int(
            next_info.get("finite_pev_count", 0)
        )
        row["mean_reliability_after"] = float(
            next_info.get("mean_reliability", 0.0)
        )
        row["uncertainty_error_after"] = next_info.get(
            "uncertainty_error",
            "",
        )

    return row


def run_pev_diagnostic(
    *,
    project_root: str | Path,
    output_directory: str | Path,
    seed: int,
    heritability: float,
    population_size: int,
    generations: int,
    phenotype_reps: int,
    maximum_steps: int,
    n_cores: int,
) -> pd.DataFrame:
    """Run the PEV action-mask diagnostic and save the step table."""
    output_path = Path(output_directory).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    env = make_low_h2_env(
        project_root=project_root,
        seed=seed,
        heritability=heritability,
        population_size=population_size,
        generations=generations,
        reps=phenotype_reps,
        n_cores=n_cores,
    )

    _, info = env.reset(seed=seed)
    rows = []

    for step in range(1, maximum_steps + 1):
        action = choose_diagnostic_action(info)
        _, reward, terminated, truncated, next_info = env.step(action)
        rows.append(
            info_row(
                step=step,
                seed=seed,
                info=info,
                action=action,
                reward=reward,
                next_info=next_info,
            )
        )
        info = next_info

        if terminated or truncated:
            break

    result = pd.DataFrame(rows)
    result.to_csv(
        output_path / "pev_mask_diagnostic.csv",
        index=False,
    )

    post_model = result[
        result["number_phenotyped_before"]
        >= env.config.minimum_training_size
    ]
    pev_enabled = int(
        post_model["mask_highest_pev_before"].sum()
    )

    print("\nPEV diagnostic complete")
    print(f"Output: {output_path / 'pev_mask_diagnostic.csv'}")
    print(f"Decision states after model-fit threshold: {len(post_model)}")
    print(f"States with Highest-PEV enabled: {pev_enabled}")

    if len(post_model) > 0 and pev_enabled == 0:
        errors = sorted(
            {
                str(value)
                for value in post_model[
                    "uncertainty_error_before"
                ].dropna()
                if str(value)
            }
        )
        print("Highest-PEV remained masked after model fitting.")
        if errors:
            print("Uncertainty errors:")
            for error in errors[:5]:
                print(f"- {error}")
    else:
        print("Highest-PEV became available after model fitting.")

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a short low-h2 Highest-PEV mask diagnostic."
    )
    parser.add_argument(
        "--output-directory",
        default="results/rl/pev_diagnostic",
    )
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
    parser.add_argument("--phenotype-reps", type=int, default=2)
    parser.add_argument("--maximum-steps", type=int, default=80)
    parser.add_argument("--n-cores", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]

    run_pev_diagnostic(
        project_root=project_root,
        output_directory=args.output_directory,
        seed=args.seed,
        heritability=args.heritability,
        population_size=args.population_size,
        generations=args.generations,
        phenotype_reps=args.phenotype_reps,
        maximum_steps=args.maximum_steps,
        n_cores=args.n_cores,
    )


if __name__ == "__main__":
    main()
