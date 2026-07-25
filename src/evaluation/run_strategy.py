"""
run_strategy.py

Shared multi-generation experiment runner.

The runner:
1. Resets the AlphaSimR breeding program.
2. Retrieves current candidate information.
3. Asks a phenotyping strategy to select candidates.
4. Sends those indices to the R simulator.
5. Records generation-level results.
6. Repeats for the requested number of generations.

The same runner can later evaluate:
- random sampling
- diversity sampling
- active learning
- reinforcement learning
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from src.baselines.base_strategy import BasePhenotypingStrategy
from src.environment.r_bridge import RBreedingBridge


@dataclass(frozen=True)
class StrategyRunConfig:
    """Configuration for one multi-generation strategy run."""

    number_of_generations: int = 20
    number_to_phenotype: int = 200
    number_of_parents: int = 20
    number_of_crosses: int = 100
    f1_per_cross: int = 1
    dh_per_f1: int = 10
    reps: int = 1
    trait: int = 1
    seed: int = 12345
    include_markers: bool = False

    def __post_init__(self) -> None:
        """Validate configuration values."""
        integer_fields = {
            "number_of_generations": self.number_of_generations,
            "number_to_phenotype": self.number_to_phenotype,
            "number_of_parents": self.number_of_parents,
            "number_of_crosses": self.number_of_crosses,
            "f1_per_cross": self.f1_per_cross,
            "dh_per_f1": self.dh_per_f1,
            "reps": self.reps,
            "trait": self.trait,
            "seed": self.seed,
        }

        for name, value in integer_fields.items():
            if isinstance(value, bool) or not isinstance(
                value,
                (int, np.integer),
            ):
                raise TypeError(f"'{name}' must be an integer.")

            if value < 1:
                raise ValueError(f"'{name}' must be at least 1.")

        if not isinstance(self.include_markers, bool):
            raise TypeError("'include_markers' must be Boolean.")


@dataclass
class StrategyRunResult:
    """Outputs from one complete multi-generation strategy run."""

    strategy_name: str
    config: StrategyRunConfig
    generation_summary: pd.DataFrame
    overall_summary: pd.DataFrame
    cycle_results: list[dict[str, Any]]
    final_candidate_data: dict[str, Any]


def run_strategy(
    *,
    bridge: RBreedingBridge,
    strategy: BasePhenotypingStrategy,
    config: StrategyRunConfig,
) -> StrategyRunResult:
    """
    Run one phenotyping strategy over several breeding generations.

    Parameters
    ----------
    bridge:
        Initialized Python-to-R breeding simulator bridge.

    strategy:
        Phenotyping strategy implementing BasePhenotypingStrategy.

    config:
        Experiment settings.

    Returns
    -------
    StrategyRunResult
        Generation summaries, overall metrics, raw cycle outputs, and
        final candidate information.
    """
    if not isinstance(bridge, RBreedingBridge):
        raise TypeError("'bridge' must be an RBreedingBridge instance.")

    if not isinstance(strategy, BasePhenotypingStrategy):
        raise TypeError(
            "'strategy' must inherit from BasePhenotypingStrategy."
        )

    if not isinstance(config, StrategyRunConfig):
        raise TypeError("'config' must be a StrategyRunConfig instance.")

    bridge.reset(seed=config.seed)
    rng = np.random.default_rng(config.seed)

    generation_rows: list[pd.DataFrame] = []
    cycle_results: list[dict[str, Any]] = []

    cumulative_phenotyping_cost = 0
    run_start = perf_counter()

    for generation in range(1, config.number_of_generations + 1):
        print(
            f"\nRunning {strategy.name}: generation "
            f"{generation} of {config.number_of_generations}..."
        )

        candidate_data = bridge.get_candidate_data(
            include_markers=config.include_markers
        )

        selection_start = perf_counter()

        selected_indices = strategy.select(
            candidate_data=candidate_data,
            number_to_phenotype=config.number_to_phenotype,
            rng=rng,
        )

        selection_seconds = perf_counter() - selection_start

        cycle_seed = config.seed + generation - 1

        cycle_start = perf_counter()

        cycle_result = bridge.step(
            selected_indices=selected_indices,
            number_of_parents=config.number_of_parents,
            number_of_crosses=config.number_of_crosses,
            f1_per_cross=config.f1_per_cross,
            dh_per_f1=config.dh_per_f1,
            reps=config.reps,
            trait=config.trait,
            seed=cycle_seed,
        )

        cycle_seconds = perf_counter() - cycle_start

        cycle_summary = cycle_result["cycle_summary"].copy()

        if len(cycle_summary) != 1:
            raise RuntimeError(
                "Each breeding cycle must return exactly one summary row."
            )

        cumulative_phenotyping_cost += int(
            cycle_summary.loc[0, "phenotyping_cost_units"]
        )

        cycle_summary.insert(0, "strategy", strategy.name)
        cycle_summary["selection_seconds"] = selection_seconds
        cycle_summary["cycle_seconds"] = cycle_seconds
        cycle_summary["cumulative_phenotyping_cost"] = (
            cumulative_phenotyping_cost
        )
        cycle_summary["run_seed"] = config.seed
        cycle_summary["cycle_seed"] = cycle_seed

        generation_rows.append(cycle_summary)
        cycle_results.append(cycle_result)

        print(
            "Prediction accuracy:",
            round(float(cycle_summary.loc[0, "prediction_accuracy"]), 3),
        )
        print(
            "Current mean genetic value:",
            round(
                float(
                    cycle_summary.loc[
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
                    cycle_summary.loc[
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
                    cycle_summary.loc[
                        0,
                        "realized_genetic_gain",
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
        generation_summary.loc[0, "population_mean_gv_before"]
    )
    final_mean_gv = float(
        generation_summary.loc[
            generation_summary.index[-1],
            "next_generation_mean_gv",
        ]
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
            "total_number_phenotyped": [
                config.number_of_generations
                * config.number_to_phenotype
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
                final_mean_gv - initial_mean_gv
            ],
            "mean_prediction_accuracy": [
                generation_summary[
                    "prediction_accuracy"
                ].mean()
            ],
            "final_genetic_variance": [
                generation_summary.loc[
                    generation_summary.index[-1],
                    "next_generation_variance_gv",
                ]
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
        include_markers=config.include_markers
    )

    return StrategyRunResult(
        strategy_name=strategy.name,
        config=config,
        generation_summary=generation_summary,
        overall_summary=overall_summary,
        cycle_results=cycle_results,
        final_candidate_data=final_candidate_data,
    )


def save_strategy_run(
    result: StrategyRunResult,
    output_directory: str | Path,
) -> dict[str, Path]:
    """
    Save summary tables from a strategy run.

    Returns paths to the created CSV files.
    """
    if not isinstance(result, StrategyRunResult):
        raise TypeError(
            "'result' must be a StrategyRunResult instance."
        )

    output_directory = Path(output_directory).expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)

    generation_path = (
        output_directory
        / f"{result.strategy_name}_generation_summary.csv"
    )
    overall_path = (
        output_directory
        / f"{result.strategy_name}_overall_summary.csv"
    )

    result.generation_summary.to_csv(
        generation_path,
        index=False,
    )
    result.overall_summary.to_csv(
        overall_path,
        index=False,
    )

    return {
        "generation_summary": generation_path,
        "overall_summary": overall_path,
    }
