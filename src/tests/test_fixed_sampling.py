"""
Tests for the fixed phenotyping baseline.
"""

from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.fixed_sampling import (
    FixedSamplingStrategy,
)


def main() -> None:
    """Run unit tests for fixed sampling."""

    candidate_data = {
        "generation": 1,
        "population_size": 1000,
        "individual_ids": [
            f"CAND_{index:04d}"
            for index in range(1, 1001)
        ],
    }

    # --------------------------------------------------------------
    # Test first-candidate rule
    # --------------------------------------------------------------

    first_strategy = FixedSamplingStrategy(
        selection_rule="first",
        sort_indices=True,
    )

    first_selection = first_strategy.select(
        candidate_data=candidate_data,
        number_to_phenotype=200,
        rng=np.random.default_rng(12345),
    )

    print("\nFirst-rule strategy:", first_strategy)
    print("First ten indices:", first_selection[:10])
    print("Last selected index:", first_selection[-1])

    assert first_selection.shape == (200,)
    assert first_selection.dtype == np.int64
    assert np.array_equal(
        first_selection,
        np.arange(200, dtype=np.int64),
    )

    # The result should not change when the random seed changes.
    repeated_first_selection = first_strategy.select(
        candidate_data=candidate_data,
        number_to_phenotype=200,
        rng=np.random.default_rng(99999),
    )

    assert np.array_equal(
        first_selection,
        repeated_first_selection,
    )

    # --------------------------------------------------------------
    # Test evenly spaced rule
    # --------------------------------------------------------------

    evenly_spaced_strategy = FixedSamplingStrategy(
        selection_rule="evenly_spaced",
        sort_indices=True,
    )

    evenly_spaced_selection = evenly_spaced_strategy.select(
        candidate_data=candidate_data,
        number_to_phenotype=200,
        rng=np.random.default_rng(12345),
    )

    print(
        "\nEvenly spaced strategy:",
        evenly_spaced_strategy,
    )
    print(
        "First ten evenly spaced indices:",
        evenly_spaced_selection[:10],
    )
    print(
        "Last ten evenly spaced indices:",
        evenly_spaced_selection[-10:],
    )

    assert evenly_spaced_selection.shape == (200,)
    assert evenly_spaced_selection.dtype == np.int64
    assert evenly_spaced_selection[0] == 0
    assert evenly_spaced_selection[-1] == 999
    assert np.unique(evenly_spaced_selection).size == 200
    assert np.all(np.diff(evenly_spaced_selection) > 0)

    # --------------------------------------------------------------
    # Test full population selection
    # --------------------------------------------------------------

    full_selection = evenly_spaced_strategy.select(
        candidate_data=candidate_data,
        number_to_phenotype=1000,
        rng=np.random.default_rng(12345),
    )

    assert np.array_equal(
        full_selection,
        np.arange(1000, dtype=np.int64),
    )

    print("\nAll fixed-sampling checks passed.")


if __name__ == "__main__":
    main()