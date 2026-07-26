"""
Run Phase 1 and Phase 2 pre-RL switching analysis.

Example
-------
python -m src.evaluation.run_pre_rl_switching
"""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

from src.evaluation.pre_rl_switching import (
    SwitchingScenario,
    run_pre_rl_switching_analysis,
    save_pre_rl_switching_analysis,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Run fixed-strategy and oracle-switching experiments before "
            "training RL."
        )
    )
    parser.add_argument(
        "--replicates",
        type=int,
        default=3,
        help="Matched replicates per scenario.",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=8,
        help="Breeding generations per replicate.",
    )
    parser.add_argument(
        "--heritabilities",
        type=float,
        nargs="+",
        default=[0.2, 0.4, 0.7],
        help="Broad-sense h2 values to test.",
    )
    parser.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=[200],
        help="Phenotyping budgets per generation.",
    )
    parser.add_argument(
        "--parents",
        type=int,
        nargs="+",
        default=[20],
        help="Numbers of selected parents.",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=30001,
        help="First scenario's base random seed.",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=None,
        help="Output directory. Defaults to results/pre_rl_switching.",
    )
    parser.add_argument(
        "--population-file",
        type=Path,
        default=Path("data/initial_candidate_population.RData"),
        help="Initial population RData file.",
    )
    return parser.parse_args()


def build_scenarios(
    args: argparse.Namespace,
) -> list[SwitchingScenario]:
    """Build the requested scenario grid."""
    scenarios: list[SwitchingScenario] = []

    for scenario_index, (
        heritability,
        budget,
        parents,
    ) in enumerate(
        product(
            args.heritabilities,
            args.budgets,
            args.parents,
        )
    ):
        name = (
            f"h2_{heritability:.2f}_budget_{budget}_"
            f"parents_{parents}"
        )
        scenarios.append(
            SwitchingScenario(
                name=name,
                heritability=heritability,
                number_of_replicates=args.replicates,
                number_of_generations=args.generations,
                number_to_phenotype=budget,
                number_of_parents=parents,
                base_seed=args.base_seed + scenario_index * 1000,
            )
        )

    return scenarios


def main() -> None:
    """Run and save the pre-RL switching analysis."""
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    output_directory = (
        args.output_directory
        if args.output_directory is not None
        else project_root / "results" / "pre_rl_switching"
    )

    result = run_pre_rl_switching_analysis(
        project_root=project_root,
        scenarios=build_scenarios(args),
        population_file=args.population_file,
    )
    paths = save_pre_rl_switching_analysis(
        result,
        output_directory,
    )

    print("\nPre-RL switching analysis completed.")
    print(
        "Total runtime seconds:",
        round(result.total_runtime_seconds, 3),
    )
    print("\nSaved files:")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
