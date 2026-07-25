"""
Test the Python-to-R breeding simulator bridge.
"""

from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Allow imports from the repository root.
sys.path.insert(0, str(PROJECT_ROOT))

from src.environment.r_bridge import RBreedingBridge


def main() -> None:
    """Run a basic bridge integration test."""

    bridge = RBreedingBridge(
        project_root=PROJECT_ROOT,
        population_file="data/initial_candidate_population.RData",
        seed=12345,
    )

    # --------------------------------------------------------------
    # Test reset
    # --------------------------------------------------------------

    reset_result = bridge.reset(seed=12345)

    print("\nR bridge initialized successfully.")
    print("Generation:", reset_result["generation"])
    print("Population size:", reset_result["population_size"])
    print(
        "First six candidate IDs:",
        reset_result["individual_ids"][:6],
    )

    assert bridge.generation == 1
    assert bridge.population_size == 1000
    assert len(bridge.get_candidate_ids()) == 1000

    # --------------------------------------------------------------
    # Test marker extraction
    # --------------------------------------------------------------

    marker_matrix = bridge.get_marker_matrix()

    print("\nMarker matrix shape:", marker_matrix.shape)
    print(
        "First candidate, first ten markers:",
        marker_matrix[0, :10],
    )

    assert marker_matrix.shape[0] == 1000
    assert marker_matrix.shape[1] == 2000
    assert np.isfinite(marker_matrix).all()

    # --------------------------------------------------------------
    # Select 200 candidates using Python zero-based indices
    # --------------------------------------------------------------

    rng = np.random.default_rng(12345)

    selected_indices = rng.choice(
        bridge.population_size,
        size=200,
        replace=False,
    )

    assert selected_indices.min() >= 0
    assert selected_indices.max() < 1000
    assert np.unique(selected_indices).size == 200

    # --------------------------------------------------------------
    # Run one R breeding cycle from Python
    # --------------------------------------------------------------

    result = bridge.step(
        selected_indices=selected_indices,
        number_of_parents=20,
        number_of_crosses=100,
        f1_per_cross=1,
        dh_per_f1=10,
        reps=1,
        trait=1,
        seed=12345,
    )

    cycle_summary = result["cycle_summary"]

    print("\nBreeding cycle completed through Python.")
    print("\nCycle summary:")
    print(cycle_summary.to_string(index=False))

    print(
        "\nNext population size:",
        result["next_population_size"],
    )

    print(
        "First six next-generation IDs:",
        result["next_individual_ids"][:6],
    )

    # --------------------------------------------------------------
    # Assertions
    # --------------------------------------------------------------

    assert result["generation"] == 1
    assert bridge.generation == 2
    assert result["next_population_size"] == 1000

    assert len(result["phenotype_table"]) == 200
    assert len(result["prediction_table"]) == 1000
    assert len(result["selection_table"]) == 20

    assert int(cycle_summary.loc[0, "population_size"]) == 1000
    assert int(cycle_summary.loc[0, "number_phenotyped"]) == 200

    assert np.isfinite(
        cycle_summary.loc[0, "prediction_accuracy"]
    )

    assert np.isfinite(
        cycle_summary.loc[0, "realized_genetic_gain"]
    )

    print("\nAll R-bridge checks passed.")


if __name__ == "__main__":
    main()