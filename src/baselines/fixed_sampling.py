"""
fixed_sampling.py

Fixed phenotyping baseline.

This strategy always selects candidates using the same predetermined
positional rule. In the current implementation, it selects the first
``number_to_phenotype`` candidates in population order.

The strategy does not use:
- marker information,
- genomic prediction,
- genetic diversity,
- prediction uncertainty,
- previous outcomes.

It represents a completely non-adaptive phenotyping policy.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from src.baselines.base_strategy import BasePhenotypingStrategy


class FixedSamplingStrategy(BasePhenotypingStrategy):
    """
    Select candidates using a fixed positional rule.

    Parameters
    ----------
    selection_rule:
        Fixed rule used to choose candidates.

        ``"first"``
            Select candidates from the beginning of the population.

        ``"evenly_spaced"``
            Select candidates at approximately equal intervals across
            the population order.

    sort_indices:
        If True, return indices in ascending order.
    """

    VALID_SELECTION_RULES = {
        "first",
        "evenly_spaced",
    }

    def __init__(
        self,
        selection_rule: str = "first",
        sort_indices: bool = True,
    ) -> None:
        super().__init__(name="fixed_sampling")

        if selection_rule not in self.VALID_SELECTION_RULES:
            raise ValueError(
                "'selection_rule' must be one of "
                f"{sorted(self.VALID_SELECTION_RULES)}."
            )

        if not isinstance(sort_indices, bool):
            raise TypeError("'sort_indices' must be Boolean.")

        self.selection_rule = selection_rule
        self.sort_indices = sort_indices

    def select(
        self,
        candidate_data: Mapping[str, Any],
        number_to_phenotype: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Select candidates using the configured fixed rule.

        The random generator is accepted to satisfy the common strategy
        interface, but this strategy does not use randomness.
        """
        population_size, number_to_phenotype = self.validate_inputs(
            candidate_data=candidate_data,
            number_to_phenotype=number_to_phenotype,
            rng=rng,
        )

        if self.selection_rule == "first":
            selected_indices = np.arange(
                number_to_phenotype,
                dtype=np.int64,
            )

        else:
            selected_indices = self._select_evenly_spaced(
                population_size=population_size,
                number_to_phenotype=number_to_phenotype,
            )

        if self.sort_indices:
            selected_indices = np.sort(selected_indices)

        return self.validate_selection(
            selected_indices=selected_indices,
            population_size=population_size,
            number_to_phenotype=number_to_phenotype,
        )

    @staticmethod
    def _select_evenly_spaced(
        population_size: int,
        number_to_phenotype: int,
    ) -> np.ndarray:
        """
        Select approximately equally spaced positions.

        Example
        -------
        For a population of 10 and a budget of 4, this may return:

        [0, 3, 6, 9]
        """
        if number_to_phenotype == population_size:
            return np.arange(
                population_size,
                dtype=np.int64,
            )

        selected_indices = np.linspace(
            start=0,
            stop=population_size - 1,
            num=number_to_phenotype,
        )

        selected_indices = np.rint(
            selected_indices
        ).astype(np.int64)

        if np.unique(selected_indices).size != number_to_phenotype:
            raise RuntimeError(
                "The evenly spaced rule produced duplicate indices."
            )

        return selected_indices

    def __repr__(self) -> str:
        """Return a readable strategy representation."""
        return (
            f"{self.__class__.__name__}("
            f"name={self.name!r}, "
            f"selection_rule={self.selection_rule!r}, "
            f"sort_indices={self.sort_indices!r})"
        )