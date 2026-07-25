"""
Test staged within-generation interaction with the R simulator.
"""

from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.environment.r_bridge import RBreedingBridge


def main() -> None:
    bridge = RBreedingBridge(
        project_root=PROJECT_ROOT,
        population_file=(
            "data/initial_candidate_population.RData"
        ),
        seed=12345,
    )

    start = bridge.start_generation()

    assert start["generation"] == 1
    assert start["population_size"] == 1000
    assert start["number_phenotyped"] == 0

    rng = np.random.default_rng(12345)

    first_batch = rng.choice(
        bridge.population_size,
        size=50,
        replace=False,
    )

    first_result = bridge.phenotype_batch(
        selected_indices=first_batch,
        reps=1,
        seed=12345,
    )

    print("\nFirst batch phenotyped:", first_result["batch_size"])
    print(
        "Total phenotyped:",
        first_result["total_phenotyped"],
    )

    assert bridge.number_currently_phenotyped == 50

    first_model = bridge.fit_current_model(
        trait=1
    )

    print(
        "Accuracy after first batch:",
        round(
            first_model["prediction_accuracy"],
            3,
        ),
    )

    unphenotyped = bridge.get_unphenotyped_indices()

    second_batch = rng.choice(
        unphenotyped,
        size=150,
        replace=False,
    )

    second_result = bridge.phenotype_batch(
        selected_indices=second_batch,
        reps=1,
        seed=12346,
    )

    assert second_result["total_phenotyped"] == 200
    assert bridge.number_currently_phenotyped == 200

    second_model = bridge.fit_current_model(
        trait=1
    )

    print(
        "Accuracy after second batch:",
        round(
            second_model["prediction_accuracy"],
            3,
        ),
    )

    final_result = bridge.finalize_generation(
        number_of_parents=20,
        number_of_crosses=100,
        f1_per_cross=1,
        dh_per_f1=10,
        trait=1,
        seed=12345,
    )

    print("\nFinal cycle summary:")
    print(
        final_result["cycle_summary"].to_string(
            index=False
        )
    )

    assert final_result["generation"] == 1
    assert bridge.generation == 2
    assert final_result["next_population_size"] == 1000

    assert len(final_result["phenotype_table"]) == 200
    assert len(final_result["prediction_table"]) == 1000
    assert len(final_result["selection_table"]) == 20

    assert (
        np.unique(
            final_result["phenotyped_indices"]
        ).size
        == 200
    )

    print("\nAll staged R-bridge checks passed.")


if __name__ == "__main__":
    main()