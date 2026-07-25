"""
Integration test for the shared multi-generation strategy runner.
"""

from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.random_sampling import RandomSamplingStrategy
from src.environment.r_bridge import RBreedingBridge
from src.evaluation.run_strategy import (
    StrategyRunConfig,
    run_strategy,
    save_strategy_run,
)


def main() -> None:
    """Run the random strategy for three generations."""

    bridge = RBreedingBridge(
        project_root=PROJECT_ROOT,
        population_file="data/initial_candidate_population.RData",
        seed=12345,
    )

    strategy = RandomSamplingStrategy(
        sort_indices=True
    )

    config = StrategyRunConfig(
        number_of_generations=3,
        number_to_phenotype=200,
        number_of_parents=20,
        number_of_crosses=100,
        f1_per_cross=1,
        dh_per_f1=10,
        reps=1,
        trait=1,
        seed=12345,
        include_markers=False,
    )

    result = run_strategy(
        bridge=bridge,
        strategy=strategy,
        config=config,
    )

    print("\nGeneration summary:")
    print(result.generation_summary.to_string(index=False))

    print("\nOverall summary:")
    print(result.overall_summary.to_string(index=False))

    assert result.strategy_name == "random_sampling"
    assert len(result.generation_summary) == 3
    assert len(result.cycle_results) == 3

    assert np.array_equal(
        result.generation_summary["generation"].to_numpy(),
        np.array([1, 2, 3]),
    )

    assert (
        result.generation_summary["number_phenotyped"] == 200
    ).all()

    assert (
        result.generation_summary[
            "number_of_selected_parents"
        ] == 20
    ).all()

    assert np.isfinite(
        result.generation_summary[
            "prediction_accuracy"
        ]
    ).all()

    assert np.isfinite(
        result.generation_summary[
            "realized_genetic_gain"
        ]
    ).all()

    assert (
        result.overall_summary.loc[
            0,
            "total_number_phenotyped",
        ]
        == 600
    )

    assert (
        result.overall_summary.loc[
            0,
            "cumulative_phenotyping_cost",
        ]
        == 600
    )

    assert (
        result.final_candidate_data["population_size"]
        == 1000
    )

    output_paths = save_strategy_run(
        result=result,
        output_directory=(
            PROJECT_ROOT
            / "results"
            / "tests"
            / "random_strategy"
        ),
    )

    assert output_paths["generation_summary"].is_file()
    assert output_paths["overall_summary"].is_file()

    print("\nSaved files:")
    for name, path in output_paths.items():
        print(f"{name}: {path}")

    print("\nAll strategy-runner checks passed.")


if __name__ == "__main__":
    main()