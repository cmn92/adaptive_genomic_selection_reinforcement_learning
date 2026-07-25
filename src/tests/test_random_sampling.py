"""
Tests for the random phenotyping baseline.
"""

from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.random_sampling import RandomSamplingStrategy


def main() -> None:
    """Run unit tests for random sampling."""

    candidate_data = {
        "generation": 1,
        "population_size": 1000,
        "individual_ids": [
            f"CAND_{index:04d}"
            for index in range(1, 1001)
        ],
    }

    strategy = RandomSamplingStrategy(
        sort_indices=True
    )

    rng_1 = np.random.default_rng(12345)

    selected_1 = strategy.select(
        candidate_data=candidate_data,
        number_to_phenotype=200,
        rng=rng_1,
    )

    print("\nStrategy:", strategy)
    print("Number selected:", selected_1.size)
    print("First ten selected indices:", selected_1[:10])
    print("Last ten selected indices:", selected_1[-10:])

    assert selected_1.shape == (200,)
    assert selected_1.dtype == np.int64
    assert selected_1.min() >= 0
    assert selected_1.max() < 1000
    assert np.unique(selected_1).size == 200
    assert np.all(np.diff(selected_1) >= 0)

    # --------------------------------------------------------------
    # Reproducibility test
    # --------------------------------------------------------------

    rng_2 = np.random.default_rng(12345)

    selected_2 = strategy.select(
        candidate_data=candidate_data,
        number_to_phenotype=200,
        rng=rng_2,
    )

    assert np.array_equal(
        selected_1,
        selected_2,
    )

    # --------------------------------------------------------------
    # Different seed should usually produce a different selection
    # --------------------------------------------------------------

    rng_3 = np.random.default_rng(54321)

    selected_3 = strategy.select(
        candidate_data=candidate_data,
        number_to_phenotype=200,
        rng=rng_3,
    )

    assert not np.array_equal(
        selected_1,
        selected_3,
    )

    # --------------------------------------------------------------
    # Selecting the entire population should return all indices
    # --------------------------------------------------------------

    full_population_strategy = RandomSamplingStrategy(
        sort_indices=True
    )

    full_selection = full_population_strategy.select(
        candidate_data=candidate_data,
        number_to_phenotype=1000,
        rng=np.random.default_rng(12345),
    )

    assert np.array_equal(
        full_selection,
        np.arange(1000, dtype=np.int64),
    )

    # --------------------------------------------------------------
    # Unsorted mode should still return a valid selection
    # --------------------------------------------------------------

    unsorted_strategy = RandomSamplingStrategy(
        sort_indices=False
    )

    unsorted_selection = unsorted_strategy.select(
        candidate_data=candidate_data,
        number_to_phenotype=200,
        rng=np.random.default_rng(12345),
    )

    assert unsorted_selection.shape == (200,)
    assert np.unique(unsorted_selection).size == 200

    print("\nAll random-sampling checks passed.")


if __name__ == "__main__":
    main()