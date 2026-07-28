"""
compare_strategies.py

Run repeated, matched comparisons of four phenotyping strategies:

- random sampling;
- fixed positional sampling;
- diversity sampling;
- PEV-based active learning.

Each replicate uses the same starting population file and the same run seed
for all four strategies. The output contains:

1. one row per strategy, replicate, and generation;
2. one row per strategy and replicate with aggregate metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Iterable

import numpy as np
import pandas as pd

from src.baselines.active_learning import ActiveLearningStrategy
from src.baselines.diversity_sampling import DiversitySamplingStrategy
from src.baselines.fixed_sampling import FixedSamplingStrategy
from src.baselines.random_sampling import RandomSamplingStrategy
from src.environment.r_bridge import RBreedingBridge
from src.evaluation.metrics import summarize_all_replicates
from src.evaluation.run_active_strategy import (
    ActiveStrategyRunConfig,
    run_active_strategy,
)
from src.evaluation.run_strategy import (
    StrategyRunConfig,
    run_strategy,
)


@dataclass(frozen=True)
class StrategyComparisonConfig:
    """Configuration for repeated matched strategy comparisons."""

    number_of_replicates: int = 20
    number_of_generations: int = 20
    number_to_phenotype: int = 200
    number_of_parents: int = 20
    number_of_crosses: int = 100
    f1_per_cross: int = 1
    dh_per_f1: int = 10
    reps: int = 1
    trait: int = 1
    snp_chip: int = 1
    active_initial_batch_size: int = 50
    n_cores: int = 1
    base_seed: int = 1001
    trait_heritability: float | None = None
    population_size: int | None = None

    def __post_init__(self) -> None:
        integer_fields = {
            "number_of_replicates": self.number_of_replicates,
            "number_of_generations": self.number_of_generations,
            "number_to_phenotype": self.number_to_phenotype,
            "number_of_parents": self.number_of_parents,
            "number_of_crosses": self.number_of_crosses,
            "f1_per_cross": self.f1_per_cross,
            "dh_per_f1": self.dh_per_f1,
            "reps": self.reps,
            "trait": self.trait,
            "snp_chip": self.snp_chip,
            "active_initial_batch_size": (
                self.active_initial_batch_size
            ),
            "n_cores": self.n_cores,
            "base_seed": self.base_seed,
        }

        if self.population_size is not None:
            integer_fields["population_size"] = self.population_size

        for name, value in integer_fields.items():
            if isinstance(value, (bool, np.bool_)):
                raise TypeError(f"'{name}' must be an integer.")
            if not isinstance(value, (int, np.integer)):
                raise TypeError(f"'{name}' must be an integer.")
            if value < 1:
                raise ValueError(f"'{name}' must be at least 1.")

        if (
            self.active_initial_batch_size
            >= self.number_to_phenotype
        ):
            raise ValueError(
                "'active_initial_batch_size' must be smaller than "
                "'number_to_phenotype'."
            )

        if (
            self.population_size is not None
            and self.number_to_phenotype > self.population_size
        ):
            raise ValueError(
                "'number_to_phenotype' cannot exceed 'population_size'."
            )

        if self.trait_heritability is not None:
            heritability = float(self.trait_heritability)
            if (
                not np.isfinite(heritability)
                or heritability <= 0.0
                or heritability > 1.0
            ):
                raise ValueError(
                    "'trait_heritability' must be greater than zero and at "
                    "most one."
                )


@dataclass
class StrategyComparisonResult:
    """Outputs from a complete repeated strategy comparison."""

    config: StrategyComparisonConfig
    generation_results: pd.DataFrame
    replicate_results: pd.DataFrame
    total_runtime_seconds: float


def _add_replicate_metadata(
    generation_summary: pd.DataFrame,
    *,
    replicate: int,
    seed: int,
) -> pd.DataFrame:
    """Add matched-replicate identifiers to generation results."""
    result = generation_summary.copy()
    result.insert(1, "replicate", int(replicate))

    if "run_seed" not in result.columns:
        result["run_seed"] = int(seed)

    return result


def compare_strategies(
    *,
    project_root: str | Path,
    population_file: str | Path = (
        "data/initial_candidate_population.RData"
    ),
    config: StrategyComparisonConfig,
    strategy_order: Iterable[str] | None = None,
) -> StrategyComparisonResult:
    """
    Run all four strategies for every matched replicate.

    Parameters
    ----------
    project_root:
        Project repository root.

    population_file:
        Initial population file relative to the project root.

    config:
        Repeated-comparison settings.

    strategy_order:
        Optional execution order. This affects runtime order only, not the
        definitions of the strategies.

    Returns
    -------
    StrategyComparisonResult
        Combined generation and replicate result tables.
    """
    if not isinstance(config, StrategyComparisonConfig):
        raise TypeError(
            "'config' must be a StrategyComparisonConfig instance."
        )

    project_root = Path(project_root).expanduser().resolve()

    strategies = {
        "random_sampling": RandomSamplingStrategy(
            sort_indices=True
        ),
        "fixed_sampling": FixedSamplingStrategy(
            selection_rule="evenly_spaced",
            sort_indices=True,
        ),
        "diversity_sampling": DiversitySamplingStrategy(
            initial_selection="centroid_farthest",
            standardize_markers=True,
            sort_indices=True,
        ),
        "active_learning_pev": ActiveLearningStrategy(
            initial_batch_size=(
                config.active_initial_batch_size
            )
        ),
    }

    if strategy_order is None:
        order = list(strategies)
    else:
        order = list(strategy_order)
        invalid = set(order).difference(strategies)
        if invalid:
            raise ValueError(
                "Unknown strategies in strategy_order: "
                + ", ".join(sorted(invalid))
            )
        if len(order) != len(set(order)):
            raise ValueError(
                "'strategy_order' cannot contain duplicates."
            )

    generation_tables: list[pd.DataFrame] = []
    start = perf_counter()

    for replicate in range(1, config.number_of_replicates + 1):
        replicate_seed = config.base_seed + replicate - 1

        print(
            f"\n=== Replicate {replicate} of "
            f"{config.number_of_replicates}; seed "
            f"{replicate_seed} ==="
        )

        for strategy_name in order:
            print(f"\n--- {strategy_name} ---")

            bridge = RBreedingBridge(
                project_root=project_root,
                population_file=population_file,
                seed=replicate_seed,
            )

            strategy = strategies[strategy_name]

            if strategy_name == "active_learning_pev":
                active_config = ActiveStrategyRunConfig(
                    number_of_generations=(
                        config.number_of_generations
                    ),
                    number_to_phenotype=(
                        config.number_to_phenotype
                    ),
                    number_of_parents=config.number_of_parents,
                    number_of_crosses=config.number_of_crosses,
                    f1_per_cross=config.f1_per_cross,
                    dh_per_f1=config.dh_per_f1,
                    reps=config.reps,
                    trait=config.trait,
                    snp_chip=config.snp_chip,
                    n_cores=config.n_cores,
                    seed=replicate_seed,
                    trait_heritability=config.trait_heritability,
                    population_size=config.population_size,
                )

                run_result = run_active_strategy(
                    bridge=bridge,
                    strategy=strategy,
                    config=active_config,
                )

            else:
                include_markers = (
                    strategy_name == "diversity_sampling"
                )

                standard_config = StrategyRunConfig(
                    number_of_generations=(
                        config.number_of_generations
                    ),
                    number_to_phenotype=(
                        config.number_to_phenotype
                    ),
                    number_of_parents=config.number_of_parents,
                    number_of_crosses=config.number_of_crosses,
                    f1_per_cross=config.f1_per_cross,
                    dh_per_f1=config.dh_per_f1,
                    reps=config.reps,
                    trait=config.trait,
                    seed=replicate_seed,
                    include_markers=include_markers,
                    trait_heritability=config.trait_heritability,
                    population_size=config.population_size,
                )

                run_result = run_strategy(
                    bridge=bridge,
                    strategy=strategy,
                    config=standard_config,
                )

            generation_tables.append(
                _add_replicate_metadata(
                    run_result.generation_summary,
                    replicate=replicate,
                    seed=replicate_seed,
                )
            )

    generation_results = pd.concat(
        generation_tables,
        ignore_index=True,
        sort=False,
    )

    generation_results = generation_results.sort_values(
        ["replicate", "strategy", "generation"]
    ).reset_index(drop=True)

    replicate_results = summarize_all_replicates(
        generation_results
    )

    total_runtime_seconds = perf_counter() - start

    return StrategyComparisonResult(
        config=config,
        generation_results=generation_results,
        replicate_results=replicate_results,
        total_runtime_seconds=total_runtime_seconds,
    )


def save_comparison_results(
    result: StrategyComparisonResult,
    output_directory: str | Path,
) -> dict[str, Path]:
    """Save raw generation and replicate result tables."""
    if not isinstance(result, StrategyComparisonResult):
        raise TypeError(
            "'result' must be a StrategyComparisonResult instance."
        )

    output_directory = Path(
        output_directory
    ).expanduser().resolve()

    raw_directory = output_directory / "raw"
    raw_directory.mkdir(parents=True, exist_ok=True)

    generation_path = (
        raw_directory / "generation_results.csv"
    )
    replicate_path = (
        raw_directory / "replicate_results.csv"
    )
    configuration_path = (
        raw_directory / "comparison_configuration.csv"
    )

    result.generation_results.to_csv(
        generation_path,
        index=False,
    )
    result.replicate_results.to_csv(
        replicate_path,
        index=False,
    )

    pd.DataFrame(
        [result.config.__dict__]
    ).assign(
        total_runtime_seconds=result.total_runtime_seconds
    ).to_csv(
        configuration_path,
        index=False,
    )

    return {
        "generation_results": generation_path,
        "replicate_results": replicate_path,
        "configuration": configuration_path,
    }


def main() -> None:
    """Run the default repeated comparison from the command line."""
    project_root = Path(__file__).resolve().parents[2]

    config = StrategyComparisonConfig(
        number_of_replicates=20,
        number_of_generations=20,
        number_to_phenotype=200,
        number_of_parents=20,
        number_of_crosses=100,
        f1_per_cross=1,
        dh_per_f1=10,
        reps=1,
        trait=1,
        snp_chip=1,
        active_initial_batch_size=50,
        n_cores=1,
        base_seed=1001,
    )

    result = compare_strategies(
        project_root=project_root,
        config=config,
    )

    paths = save_comparison_results(
        result,
        project_root / "results" / "evaluation",
    )

    print("\nComparison completed.")
    print(
        "Total runtime seconds:",
        round(result.total_runtime_seconds, 3),
    )

    print("\nSaved files:")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
