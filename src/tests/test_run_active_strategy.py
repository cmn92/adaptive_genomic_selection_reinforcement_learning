"""
test_run_active_strategy.py

Integration test for the multi-generation active-learning runner.

This test runs the formal PEV-based active-learning strategy for three
breeding generations and verifies that:

- each generation uses the full phenotyping budget;
- the initial and uncertainty batches have the expected sizes;
- prediction and breeding metrics remain finite;
- the final population contains 1,000 candidates;
- output CSV files are created successfully.
"""

from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.active_learning import ActiveLearningStrategy
from src.environment.r_bridge import RBreedingBridge
from src.evaluation.run_active_strategy import (
    ActiveStrategyRunConfig,
    run_active_strategy,
    save_active_strategy_run,
)


def main() -> None:
    """Run a three-generation active-learning integration test."""

    bridge = RBreedingBridge(
        project_root=PROJECT_ROOT,
        population_file="data/initial_candidate_population.RData",
        seed=12345,
    )

    strategy = ActiveLearningStrategy(
        initial_batch_size=50,
    )

    config = ActiveStrategyRunConfig(
        number_of_generations=3,
        number_to_phenotype=200,
        number_of_parents=20,
        number_of_crosses=100,
        f1_per_cross=1,
        dh_per_f1=10,
        reps=1,
        trait=1,
        snp_chip=1,
        n_cores=1,
        seed=12345,
    )

    result = run_active_strategy(
        bridge=bridge,
        strategy=strategy,
        config=config,
    )

    print("\nGeneration summary:")
    print(result.generation_summary.to_string(index=False))

    print("\nOverall summary:")
    print(result.overall_summary.to_string(index=False))

    assert result.strategy_name == "active_learning_pev"
    assert len(result.generation_summary) == 3
    assert len(result.generation_results) == 3

    assert np.array_equal(
        result.generation_summary["generation"].to_numpy(),
        np.array([1, 2, 3]),
    )

    assert (
        result.generation_summary["number_phenotyped"] == 200
    ).all()

    assert (
        result.generation_summary["initial_batch_size"] == 50
    ).all()

    assert (
        result.generation_summary["uncertainty_batch_size"] == 150
    ).all()

    assert (
        result.generation_summary["training_population_size"] == 200
    ).all()

    assert (
        result.generation_summary["number_of_selected_parents"] == 20
    ).all()

    assert np.isfinite(
        result.generation_summary["initial_model_accuracy"]
    ).all()

    assert np.isfinite(
        result.generation_summary["final_model_accuracy"]
    ).all()

    assert np.isfinite(
        result.generation_summary["prediction_accuracy"]
    ).all()

    assert np.isfinite(
        result.generation_summary["realized_genetic_gain"]
    ).all()

    assert np.isfinite(
        result.generation_summary["next_generation_variance_gv"]
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
        result.overall_summary.loc[
            0,
            "initial_batch_size",
        ]
        == 50
    )

    assert (
        result.overall_summary.loc[
            0,
            "uncertainty_batch_size",
        ]
        == 150
    )

    assert np.isfinite(
        result.overall_summary.loc[
            0,
            "total_realized_genetic_gain",
        ]
    )

    assert np.isfinite(
        result.overall_summary.loc[
            0,
            "variance_retention",
        ]
    )

    assert (
        result.final_candidate_data["population_size"]
        == 1000
    )

    for generation_result in result.generation_results:
        assert generation_result.initial_indices.shape == (50,)
        assert generation_result.uncertainty_indices.shape == (150,)
        assert generation_result.all_phenotyped_indices.shape == (200,)

        assert np.unique(
            generation_result.initial_indices
        ).size == 50

        assert np.unique(
            generation_result.uncertainty_indices
        ).size == 150

        assert np.unique(
            generation_result.all_phenotyped_indices
        ).size == 200

        assert np.intersect1d(
            generation_result.initial_indices,
            generation_result.uncertainty_indices,
        ).size == 0

        assert len(generation_result.phenotype_table) == 200
        assert len(generation_result.prediction_table) == 1000
        assert len(generation_result.selection_table) == 20
        assert generation_result.next_population_size == 1000

    output_paths = save_active_strategy_run(
        result=result,
        output_directory=(
            PROJECT_ROOT
            / "results"
            / "tests"
            / "active_learning"
        ),
    )

    assert output_paths["generation_summary"].is_file()
    assert output_paths["overall_summary"].is_file()
    assert output_paths["selected_candidates"].is_file()

    print("\nSaved files:")
    for name, path in output_paths.items():
        print(f"{name}: {path}")

    print("\nAll active-strategy-runner checks passed.")


if __name__ == "__main__":
    main()
