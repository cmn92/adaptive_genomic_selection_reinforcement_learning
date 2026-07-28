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
    SwitchingPolicyConfig,
    SwitchingScenario,
    run_pre_rl_switching_analysis,
    save_pre_rl_switching_analysis,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Run fixed-strategy and empirical switching experiments before "
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
        nargs="+",
        default=[5, 8, 12],
        help="Breeding-generation counts to test.",
    )
    parser.add_argument(
        "--scenario-mode",
        choices=[
            "compact",
            "full-factorial",
            "confirmatory-low-h2",
        ],
        default="compact",
        help=(
            "compact varies one factor at a time around a base scenario; "
            "full-factorial runs every combination; confirmatory-low-h2 "
            "focuses on promising low-heritability conditions."
        ),
    )
    parser.add_argument(
        "--heritabilities",
        type=float,
        nargs="+",
        default=[0.05, 0.1, 0.2, 0.4, 0.7],
        help="Broad-sense h2 values to test.",
    )
    parser.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=[75, 100, 200],
        help="Phenotyping budgets per generation.",
    )
    parser.add_argument(
        "--population-sizes",
        type=int,
        nargs="+",
        default=[500, 1000],
        help="Candidate population sizes to test.",
    )
    parser.add_argument(
        "--parents",
        type=int,
        nargs="+",
        default=[10, 20, 40],
        help="Numbers of selected parents.",
    )
    parser.add_argument(
        "--diversity-losses",
        choices=["weak", "standard", "strong"],
        nargs="+",
        default=["weak", "standard", "strong"],
        help=(
            "Family-bottleneck settings. Weak uses many crosses and "
            "fewer DH per cross; strong uses fewer crosses and more DH "
            "per cross at the same population size."
        ),
    )
    parser.add_argument(
        "--active-initial-batch-size",
        type=int,
        default=50,
        help="Initial diverse batch size for staged model-based strategies.",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=30001,
        help="First scenario's base random seed.",
    )
    parser.add_argument(
        "--switching-policy",
        choices=[
            "one_step_greedy",
            "averaged_one_step",
            "rollout_to_end",
        ],
        default="one_step_greedy",
        help=(
            "Empirical switching policy for Phase 2. "
            "rollout_to_end scores each first action by continuing with "
            "the best fixed strategy."
        ),
    )
    parser.add_argument(
        "--action-repeats",
        type=int,
        default=1,
        help=(
            "Number of stochastic draws per candidate action for "
            "averaged_one_step or rollout_to_end scoring."
        ),
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


def breeding_design_for(
    *,
    population_size: int,
    diversity_loss: str,
) -> tuple[int, int]:
    """Return number_of_crosses and dh_per_f1 for a bottleneck level."""
    dh_by_loss = {
        "weak": 5,
        "standard": 10,
        "strong": 20,
    }
    dh_per_f1 = dh_by_loss[diversity_loss]

    if population_size % dh_per_f1 != 0:
        raise ValueError(
            "Population size must be divisible by the DH-per-F1 value "
            f"for diversity_loss={diversity_loss!r}."
        )

    return population_size // dh_per_f1, dh_per_f1


def active_initial_batch_size(
    *,
    requested_size: int,
    budget: int,
) -> int:
    """Keep the staged initial batch valid for smaller budgets."""
    if budget < 3:
        raise ValueError("Phenotyping budget must be at least 3.")
    return max(2, min(requested_size, budget - 1))


def scenario_name(
    *,
    heritability: float,
    budget: int,
    population_size: int,
    parents: int,
    generations: int,
    diversity_loss: str,
) -> str:
    """Create a stable scenario identifier."""
    return (
        f"h2_{heritability:.2f}_budget_{budget}_"
        f"pop_{population_size}_parents_{parents}_"
        f"gen_{generations}_loss_{diversity_loss}"
    )


def make_scenario(
    *,
    args: argparse.Namespace,
    scenario_index: int,
    heritability: float,
    budget: int,
    population_size: int,
    parents: int,
    generations: int,
    diversity_loss: str,
) -> SwitchingScenario:
    """Build one scenario with a population-size preserving design."""
    if budget > population_size:
        raise ValueError(
            "Phenotyping budget cannot exceed population size "
            f"for {scenario_name(heritability=heritability, budget=budget, population_size=population_size, parents=parents, generations=generations, diversity_loss=diversity_loss)}."
        )

    if parents > budget:
        raise ValueError(
            "Number of parents cannot exceed phenotyping budget "
            f"for {scenario_name(heritability=heritability, budget=budget, population_size=population_size, parents=parents, generations=generations, diversity_loss=diversity_loss)}."
        )

    number_of_crosses, dh_per_f1 = breeding_design_for(
        population_size=population_size,
        diversity_loss=diversity_loss,
    )

    return SwitchingScenario(
        name=scenario_name(
            heritability=heritability,
            budget=budget,
            population_size=population_size,
            parents=parents,
            generations=generations,
            diversity_loss=diversity_loss,
        ),
        heritability=heritability,
        population_size=population_size,
        diversity_loss=diversity_loss,
        number_of_replicates=args.replicates,
        number_of_generations=generations,
        number_to_phenotype=budget,
        number_of_parents=parents,
        number_of_crosses=number_of_crosses,
        dh_per_f1=dh_per_f1,
        active_initial_batch_size=active_initial_batch_size(
            requested_size=args.active_initial_batch_size,
            budget=budget,
        ),
        base_seed=args.base_seed + scenario_index * 1000,
    )


def preferred_value(
    values: list,
    preferred,
):
    """Use a preferred compact-grid value when available."""
    return preferred if preferred in values else values[0]


def build_scenarios(
    args: argparse.Namespace,
) -> list[SwitchingScenario]:
    """Build the requested scenario grid."""
    scenario_specs: list[tuple[float, int, int, int, int, str]]

    base = (
        preferred_value(args.heritabilities, 0.4),
        preferred_value(args.budgets, 200),
        preferred_value(args.population_sizes, 1000),
        preferred_value(args.parents, 20),
        preferred_value(args.generations, 8),
        preferred_value(args.diversity_losses, "standard"),
    )

    if args.scenario_mode == "full-factorial":
        scenario_specs = list(
            product(
                args.heritabilities,
                args.budgets,
                args.population_sizes,
                args.parents,
                args.generations,
                args.diversity_losses,
            )
        )
    elif args.scenario_mode == "confirmatory-low-h2":
        low_h2 = [
            value
            for value in args.heritabilities
            if value <= 0.2
        ] or [0.05, 0.1, 0.2]
        budgets = [
            value
            for value in args.budgets
            if value in {75, 200}
        ] or [75, 200]
        populations = [
            value
            for value in args.population_sizes
            if value in {500, 1000}
        ] or [500, 1000]
        parents = [
            value
            for value in args.parents
            if value in {20, 40}
        ] or [20, 40]
        generations = [preferred_value(args.generations, 8)]
        losses = [
            value
            for value in args.diversity_losses
            if value in {"standard", "strong"}
        ] or ["standard"]
        scenario_specs = list(
            product(
                low_h2,
                budgets,
                populations,
                parents,
                generations,
                losses,
            )
        )
    else:
        specs: list[tuple[float, int, int, int, int, str]] = []

        for heritability in args.heritabilities:
            specs.append(
                (
                    heritability,
                    base[1],
                    base[2],
                    base[3],
                    base[4],
                    base[5],
                )
            )

        for budget in args.budgets:
            specs.append((base[0], budget, base[2], base[3], base[4], base[5]))

        for population_size in args.population_sizes:
            specs.append(
                (
                    base[0],
                    base[1],
                    population_size,
                    base[3],
                    base[4],
                    base[5],
                )
            )

        for parents in args.parents:
            specs.append((base[0], base[1], base[2], parents, base[4], base[5]))

        for generations in args.generations:
            specs.append(
                (
                    base[0],
                    base[1],
                    base[2],
                    base[3],
                    generations,
                    base[5],
                )
            )

        for diversity_loss in args.diversity_losses:
            specs.append(
                (
                    base[0],
                    base[1],
                    base[2],
                    base[3],
                    base[4],
                    diversity_loss,
                )
            )

        scenario_specs = list(dict.fromkeys(specs))

    scenarios = [
        make_scenario(
            args=args,
            scenario_index=index,
            heritability=heritability,
            budget=budget,
            population_size=population_size,
            parents=parents,
            generations=generations,
            diversity_loss=diversity_loss,
        )
        for index, (
            heritability,
            budget,
            population_size,
            parents,
            generations,
            diversity_loss,
        ) in enumerate(scenario_specs)
    ]

    if args.scenario_mode == "confirmatory-low-h2":
        scenarios = [
            SwitchingScenario(
                **{
                    **scenario.__dict__,
                    "number_of_replicates": max(
                        args.replicates,
                        20,
                    ),
                }
            )
            for scenario in scenarios
        ]

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
        switching_policy=SwitchingPolicyConfig(
            policy=args.switching_policy,
            action_repeats=args.action_repeats,
        ),
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
