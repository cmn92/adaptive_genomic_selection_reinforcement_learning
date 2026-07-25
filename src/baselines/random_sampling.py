"""
random_sampling.py

Random phenotyping baseline.

This strategy selects a fixed number of candidates uniformly at random
without replacement.

It serves as the simplest baseline for adaptive phenotyping. Any more
advanced strategy should be compared against it under the same population,
phenotyping budget, breeding design, and random-seed structure.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from src.baselines.base_strategy import BasePhenotypingStrategy


class RandomSamplingStrategy(BasePhenotypingStrategy):
    """
    Select candidates uniformly at random without replacement.

    The strategy uses only the current population size. It does not use
    marker data, predicted breeding values, pedigree, family information,
    or prediction uncertainty.
    """

    def __init__(self, sort_indices: bool = True) -> None:
        """
        Initialize the random-sampling strategy.

        Parameters
        ----------
        sort_indices:
            If True, return the selected indices in ascending order.

            Sorting does not change which candidates are selected. It only
            makes logs, tests, and comparisons easier to read.
        """
        super().__init__(name="random_sampling")

        if not isinstance(sort_indices, bool):
            raise TypeError("'sort_indices' must be a Boolean value.")

        self.sort_indices = sort_indices

    def select(
        self,
        candidate_data: Mapping[str, Any],
        number_to_phenotype: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Select candidates uniformly at random.

        Parameters
        ----------
        candidate_data:
            Information about the current candidate population. Only the
            ``population_size`` field is required.

        number_to_phenotype:
            Number of candidates to select.

        rng:
            NumPy random-number generator supplied by the experiment
            runner.

        Returns
        -------
        numpy.ndarray
            Unique zero-based candidate indices.
        """
        population_size, number_to_phenotype = self.validate_inputs(
            candidate_data=candidate_data,
            number_to_phenotype=number_to_phenotype,
            rng=rng,
        )

        selected_indices = rng.choice(
            population_size,
            size=number_to_phenotype,
            replace=False,
        ).astype(np.int64)

        if self.sort_indices:
            selected_indices = np.sort(selected_indices)

        return self.validate_selection(
            selected_indices=selected_indices,
            population_size=population_size,
            number_to_phenotype=number_to_phenotype,
        )

    def __repr__(self) -> str:
        """Return a readable strategy representation."""
        return (
            f"{self.__class__.__name__}("
            f"name={self.name!r}, "
            f"sort_indices={self.sort_indices!r})"
        )