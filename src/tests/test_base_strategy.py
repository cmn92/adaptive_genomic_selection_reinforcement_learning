"""
Tests for the shared phenotyping-strategy interface.
"""

from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.base_strategy import (
    BasePhenotypingStrategy,
    StrategyValidationError,
)


class FirstCandidatesStrategy(BasePhenotypingStrategy):
    """
    Small concrete strategy used only to test the abstract base class.
    """

    def __init__(self) -> None:
        super().__init__(name="first_candidates")

    def select(
        self,
        candidate_data,
        number_to_phenotype,
        rng,
    ) -> np.ndarray:
        population_size, number_to_phenotype = self.validate_inputs(
            candidate_data=candidate_data,
            number_to_phenotype=number_to_phenotype,
            rng=rng,
        )

        selected_indices = np.arange(
            number_to_phenotype,
            dtype=np.int64,
        )

        return self.validate_selection(
            selected_indices=selected_indices,
            population_size=population_size,
            number_to_phenotype=number_to_phenotype,
        )


def main() -> None:
    """Run basic unit tests."""

    candidate_data = {
        "generation": 1,
        "population_size": 1000,
        "individual_ids": [
            f"CAND_{index:04d}"
            for index in range(1, 1001)
        ],
    }

    rng = np.random.default_rng(12345)
    strategy = FirstCandidatesStrategy()

    selected_indices = strategy.select(
        candidate_data=candidate_data,
        number_to_phenotype=200,
        rng=rng,
    )

    print("\nStrategy:", strategy)
    print("Number selected:", selected_indices.size)
    print("First six selected indices:", selected_indices[:6])
    print("Last selected index:", selected_indices[-1])

    assert strategy.name == "first_candidates"
    assert selected_indices.dtype == np.int64
    assert selected_indices.shape == (200,)
    assert selected_indices[0] == 0
    assert selected_indices[-1] == 199
    assert np.unique(selected_indices).size == 200

    # Test a strategy returning duplicate indices.
    try:
        strategy.validate_selection(
            selected_indices=np.array([0, 0, 1]),
            population_size=1000,
            number_to_phenotype=3,
        )
    except StrategyValidationError:
        pass
    else:
        raise AssertionError(
            "Duplicate indices should have raised StrategyValidationError."
        )

    # Test incorrect selection size.
    try:
        strategy.validate_selection(
            selected_indices=np.array([0, 1]),
            population_size=1000,
            number_to_phenotype=3,
        )
    except StrategyValidationError:
        pass
    else:
        raise AssertionError(
            "Incorrect selection size should have raised "
            "StrategyValidationError."
        )

    # Test an out-of-range index.
    try:
        strategy.validate_selection(
            selected_indices=np.array([0, 1, 1000]),
            population_size=1000,
            number_to_phenotype=3,
        )
    except StrategyValidationError:
        pass
    else:
        raise AssertionError(
            "Out-of-range indices should have raised "
            "StrategyValidationError."
        )

    print("\nAll base-strategy checks passed.")


if __name__ == "__main__":
    main()