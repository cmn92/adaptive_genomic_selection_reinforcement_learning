"""
run_active_strategy.py

Shared multi-generation runner for the staged PEV-based active-learning
phenotyping strategy.

This runner is separate from run_strategy.py because ActiveLearningStrategy
does not select all candidates in one call. Instead, each generation has a
staged workflow:

1. Select and phenotype an initial diverse batch.
2. Fit an initial genomic prediction model.
3. Calculate prediction error variance (PEV).
4. Select and phenotype the most uncertain remaining candidates.
5. Refit the final genomic prediction model.
6. Select parents and create the next generation.

The runner records generation-level and overall results in the same general
format used for the other baseline strategies.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from src.baselines.active_learning import (
    ActiveLearningGenerationResult,
    ActiveLearningStrategy,
)
from src.environment.r_bridge import RBreedingBridge


@dataclass(frozen=True)
class ActiveStrategyRunConfig:
    """Configuration for one multi-generation active-learning run."""

    number_of_generations: int = 20
    number_to_phenotype: int = 200
    number_of_parents: int = 20
    number_of_crosses: int = 100
    f1_per_cross: int = 1
    dh_per_f1: int = 10
    reps: int = 1
    trait: int = 1
    snp_chip: int = 1
    n_cores: int = 1
    seed: int = 12345

    def __post_init__(self) -> None:
        """Validate all configuration values."""
        integer_fields = {
            "number_of_generations": self.number_of_generations,
            "number_to_phenotype": self.number_to_phenotype,
            "number_of_parents": self.number_of_parents,
            "number_of_crosses": self.number_of_crosses,
            "f1_per_cross": self.f1_per_cross,
            "dh_per_f1": self.dh_per_f1,
            "reps": self.reps,
            "trait": self.trait,
            "snp_chip": self.snp_chip,
            "n_cores": self.n_cores,
            "seed": self.seed,
        }

        for name, value in integer_fields.items():
            if isinstance(value, (bool, np.bool_)):
                raise TypeError(f"'{name}' must be an integer.")

            if not isinstance(value, (int, np.integer)):
                raise TypeError(f"'{name}' must be an integer.")

            if value < 1:
                raise ValueError(f"'{name}' must be at least 1.")

        if self.number_of_parents > self.number_to_phenotype:
            raise ValueError(
                "'number_of_parents' cannot exceed "
                "'number_to_phenotype'."
            )


@dataclass
class ActiveStrategyRunResult:
    """Outputs from one multi-generation active-learning run."""

    strategy_name: str
    config: ActiveStrategyRunConfig
    generation_summary: pd.DataFrame
    overall_summary: pd.DataFrame
    generation_results: list[ActiveLearningGenerationResult]
    final_candidate_data: dict[str, Any]


def run_active_strategy(
    *,
    bridge: RBreedingBridge,
    strategy: ActiveLearningStrategy,
    config: ActiveStrategyRunConfig,
) -> ActiveStrategyRunResult:
    """
    Run the active-learning strategy across multiple breeding generations.

    Parameters
    ----------
    bridge:
        Initialized Python-to-R breeding simulator bridge.

    strategy:
        PEV-based active-learning strategy.

    config:
        Multi-generation experiment configuration.

    Returns
    -------
    ActiveStrategyRunResult
        Generation-level summaries, aggregate results, raw generation
        outputs, and final candidate information.
    """
    if not isinstance(bridge, RBreedingBridge):
        raise TypeError("'bridge' must be an RBreedingBridge instance.")

    if not isinstance(strategy, ActiveLearningStrategy):
        raise TypeError(
            "'strategy' must be an ActiveLearningStrategy instance."
        )

    if not isinstance(config, ActiveStrategyRunConfig):
        raise TypeError(
            "'config' must be an ActiveStrategyRunConfig instance."
        )

    if strategy.initial_batch_size >= config.number_to_phenotype:
        raise ValueError(
            "The strategy's initial batch size must be smaller than "
            "the total phenotyping budget."
        )

    bridge.reset(seed=config.seed)
    rng = np.random.default_rng(config.seed)

    generation_rows: list[pd.DataFrame] = []
    generation_results: list[ActiveLearningGenerationResult] = []

    cumulative_phenotyping_cost = 0
    run_start = perf_counter()

    for generation in range(1, config.number_of_generations + 1):
        print(
            f"\nRunning {strategy.name}: generation "
            f"{generation} of {config.number_of_generations}..."
        )

        generation_seed = config.seed + generation - 1

        generation_result = strategy.run_generation(
            bridge=bridge,
            number_to_phenotype=config.number_to_phenotype,
            rng=rng,
            number_of_parents=config.number_of_parents,
            number_of_crosses=config.number_of_crosses,
            f1_per_cross=config.f1_per_cross,
            dh_per_f1=config.dh_per_f1,
            reps=config.reps,
            trait=config.trait,
            snp_chip=config.snp_chip,
            n_cores=config.n_cores,
            seed=generation_seed,
        )

        generation_summary = generation_result.cycle_summary.copy()

        if len(generation_summary) != 1:
            raise RuntimeError(
                "Each active-learning generation must return exactly "
                "one cycle-summary row."
            )

        expected_generation = generation

        actual_generation = int(
            generation_summary.loc[0, "generation"]
        )

        if actual_generation != expected_generation:
            raise RuntimeError(
                "The active-learning generation number does not match "
                f"the runner. Expected {expected_generation}, obtained "
                f"{actual_generation}."
            )

        generation_cost = int(
            generation_summary.loc[
                0,
                "phenotyping_cost_units",
            ]
        )

        cumulative_phenotyping_cost += generation_cost

        generation_summary["cumulative_phenotyping_cost"] = (
            cumulative_phenotyping_cost
        )
        generation_summary["run_seed"] = config.seed
        generation_summary["cycle_seed"] = generation_seed

        generation_rows.append(generation_summary)
        generation_results.append(generation_result)

        print(
            "Initial model accuracy:",
            round(
                float(
                    generation_summary.loc[
                        0,
                        "initial_model_accuracy",
                    ]
                ),
                3,
            ),
        )

        print(
            "Final model accuracy:",
            round(
                float(
                    generation_summary.loc[
                        0,
                        "final_model_accuracy",
                    ]
                ),
                3,
            ),
        )

        print(
            "Current mean genetic value:",
            round(
                float(
                    generation_summary.loc[
                        0,
                        "population_mean_gv_before",
                    ]
                ),
                3,
            ),
        )

        print(
            "Next-generation mean genetic value:",
            round(
                float(
                    generation_summary.loc[
                        0,
                        "next_generation_mean_gv",
                    ]
                ),
                3,
            ),
        )

        print(
            "Realized genetic gain:",
            round(
                float(
                    generation_summary.loc[
                        0,
                        "realized_genetic_gain",
                    ]
                ),
                3,
            ),
        )

        print(
            "Genetic variance:",
            round(
                float(
                    generation_summary.loc[
                        0,
                        "next_generation_variance_gv",
                    ]
                ),
                3,
            ),
        )

    generation_summary = pd.concat(
        generation_rows,
        ignore_index=True,
    )

    total_runtime_seconds = perf_counter() - run_start

    initial_mean_gv = float(
        generation_summary.loc[
            0,
            "population_mean_gv_before",
        ]
    )

    final_row_index = generation_summary.index[-1]

    final_mean_gv = float(
        generation_summary.loc[
            final_row_index,
            "next_generation_mean_gv",
        ]
    )

    initial_genetic_variance = float(
        generation_summary.loc[
            0,
            "population_variance_gv_before",
        ]
    )

    final_genetic_variance = float(
        generation_summary.loc[
            final_row_index,
            "next_generation_variance_gv",
        ]
    )

    if initial_genetic_variance > 0:
        variance_retention = (
            final_genetic_variance
            / initial_genetic_variance
        )
    else:
        variance_retention = np.nan

    total_realized_genetic_gain = (
        final_mean_gv - initial_mean_gv
    )

    total_number_phenotyped = int(
        generation_summary[
            "number_phenotyped"
        ].sum()
    )

    gain_per_100_phenotypes = (
        total_realized_genetic_gain
        / total_number_phenotyped
        * 100
    )

    overall_summary = pd.DataFrame(
        {
            "strategy": [strategy.name],
            "number_of_generations": [
                config.number_of_generations
            ],
            "number_phenotyped_per_generation": [
                config.number_to_phenotype
            ],
            "initial_batch_size": [
                strategy.initial_batch_size
            ],
            "uncertainty_batch_size": [
                config.number_to_phenotype
                - strategy.initial_batch_size
            ],
            "total_number_phenotyped": [
                total_number_phenotyped
            ],
            "cumulative_phenotyping_cost": [
                cumulative_phenotyping_cost
            ],
            "initial_mean_genetic_value": [
                initial_mean_gv
            ],
            "final_mean_genetic_value": [
                final_mean_gv
            ],
            "total_realized_genetic_gain": [
                total_realized_genetic_gain
            ],
            "mean_gain_per_generation": [
                generation_summary[
                    "realized_genetic_gain"
                ].mean()
            ],
            "gain_per_100_phenotypes": [
                gain_per_100_phenotypes
            ],
            "mean_initial_model_accuracy": [
                generation_summary[
                    "initial_model_accuracy"
                ].mean()
            ],
            "mean_final_model_accuracy": [
                generation_summary[
                    "final_model_accuracy"
                ].mean()
            ],
            "final_prediction_accuracy": [
                generation_summary.loc[
                    final_row_index,
                    "final_model_accuracy",
                ]
            ],
            "initial_genetic_variance": [
                initial_genetic_variance
            ],
            "final_genetic_variance": [
                final_genetic_variance
            ],
            "variance_retention": [
                variance_retention
            ],
            "mean_selection_seconds": [
                generation_summary[
                    "selection_seconds"
                ].mean()
            ],
            "mean_cycle_seconds": [
                generation_summary[
                    "cycle_seconds"
                ].mean()
            ],
            "total_runtime_seconds": [
                total_runtime_seconds
            ],
            "seed": [config.seed],
        }
    )

    final_candidate_data = bridge.get_candidate_data(
        include_markers=False
    )

    return ActiveStrategyRunResult(
        strategy_name=strategy.name,
        config=config,
        generation_summary=generation_summary,
        overall_summary=overall_summary,
        generation_results=generation_results,
        final_candidate_data=final_candidate_data,
    )


def save_active_strategy_run(
    result: ActiveStrategyRunResult,
    output_directory: str | Path,
) -> dict[str, Path]:
    """
    Save active-learning summary tables and selected-candidate records.

    Parameters
    ----------
    result:
        Completed multi-generation active-learning result.

    output_directory:
        Directory where CSV files will be created.

    Returns
    -------
    dict[str, Path]
        Paths to the saved files.
    """
    if not isinstance(result, ActiveStrategyRunResult):
        raise TypeError(
            "'result' must be an ActiveStrategyRunResult instance."
        )

    output_directory = Path(
        output_directory
    ).expanduser().resolve()

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    generation_path = (
        output_directory
        / f"{result.strategy_name}_generation_summary.csv"
    )

    overall_path = (
        output_directory
        / f"{result.strategy_name}_overall_summary.csv"
    )

    selection_records: list[pd.DataFrame] = []

    for generation_result in result.generation_results:
        initial_table = pd.DataFrame(
            {
                "strategy": result.strategy_name,
                "generation": generation_result.generation,
                "selection_stage": "initial_diversity",
                "python_index": generation_result.initial_indices,
                "population_index": (
                    generation_result.initial_indices + 1
                ),
            }
        )

        uncertainty_table = pd.DataFrame(
            {
                "strategy": result.strategy_name,
                "generation": generation_result.generation,
                "selection_stage": "highest_pev",
                "python_index": (
                    generation_result.uncertainty_indices
                ),
                "population_index": (
                    generation_result.uncertainty_indices + 1
                ),
            }
        )

        selection_records.extend(
            [
                initial_table,
                uncertainty_table,
            ]
        )

    selected_candidates = pd.concat(
        selection_records,
        ignore_index=True,
    )

    selections_path = (
        output_directory
        / f"{result.strategy_name}_selected_candidates.csv"
    )

    result.generation_summary.to_csv(
        generation_path,
        index=False,
    )

    result.overall_summary.to_csv(
        overall_path,
        index=False,
    )

    selected_candidates.to_csv(
        selections_path,
        index=False,
    )

    return {
        "generation_summary": generation_path,
        "overall_summary": overall_path,
        "selected_candidates": selections_path,
    }
