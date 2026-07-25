"""
Integration test for the formal PEV-based active-learning baseline.
"""

from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.active_learning import (
    ActiveLearningStrategy,
)
from src.environment.r_bridge import RBreedingBridge


def main() -> None:
    """Run one complete active-learning breeding generation."""

    bridge = RBreedingBridge(
        project_root=PROJECT_ROOT,
        population_file=(
            "data/initial_candidate_population.RData"
        ),
        seed=12345,
    )

    strategy = ActiveLearningStrategy(
        initial_batch_size=50
    )

    result = strategy.run_generation(
        bridge=bridge,
        number_to_phenotype=200,
        rng=np.random.default_rng(12345),
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

    print("\nActive-learning generation completed.")
    print("Strategy:", strategy)

    print(
        "\nInitial diverse batch size:",
        result.initial_indices.size,
    )
    print(
        "PEV-selected batch size:",
        result.uncertainty_indices.size,
    )
    print(
        "Total phenotyped:",
        result.all_phenotyped_indices.size,
    )

    print(
        "Initial model accuracy:",
        round(result.initial_model_accuracy, 3),
    )
    print(
        "Final model accuracy:",
        round(result.final_model_accuracy, 3),
    )

    print("\nMost uncertain selected candidates:")
    selected_pev_rows = (
        result.uncertainty_table[
            result.uncertainty_table[
                "population_index"
            ].isin(
                result.uncertainty_indices + 1
            )
        ]
        .sort_values(
            "prediction_error_variance",
            ascending=False,
        )
        .head(10)
    )

    print(
        selected_pev_rows[
            [
                "population_index",
                "individual_id",
                "prediction_error_variance",
                "reliability",
            ]
        ].to_string(index=False)
    )

    print("\nCycle summary:")
    print(
        result.cycle_summary.to_string(
            index=False
        )
    )

    assert result.generation == 1
    assert bridge.generation == 2

    assert result.initial_indices.shape == (50,)
    assert result.uncertainty_indices.shape == (150,)
    assert result.all_phenotyped_indices.shape == (200,)

    assert np.unique(
        result.initial_indices
    ).size == 50

    assert np.unique(
        result.uncertainty_indices
    ).size == 150

    assert np.unique(
        result.all_phenotyped_indices
    ).size == 200

    assert np.intersect1d(
        result.initial_indices,
        result.uncertainty_indices,
    ).size == 0

    assert len(result.phenotype_table) == 200
    assert len(result.prediction_table) == 1000
    assert len(result.selection_table) == 20
    assert result.next_population_size == 1000

    assert np.isfinite(
        result.initial_model_accuracy
    )
    assert np.isfinite(
        result.final_model_accuracy
    )

    assert (
        result.cycle_summary.loc[
            0,
            "number_phenotyped",
        ]
        == 200
    )

    assert (
        result.cycle_summary.loc[
            0,
            "initial_batch_size",
        ]
        == 50
    )

    assert (
        result.cycle_summary.loc[
            0,
            "uncertainty_batch_size",
        ]
        == 150
    )

    selected_pevs = result.uncertainty_table[
        result.uncertainty_table[
            "population_index"
        ].isin(
            result.uncertainty_indices + 1
        )
    ]["prediction_error_variance"]

    unselected_pevs = result.uncertainty_table[
        ~result.uncertainty_table[
            "population_index"
        ].isin(
            result.uncertainty_indices + 1
        )
    ]["prediction_error_variance"]

    assert selected_pevs.min() >= unselected_pevs.max()

    print("\nAll active-learning checks passed.")


if __name__ == "__main__":
    main()