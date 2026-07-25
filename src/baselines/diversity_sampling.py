"""
diversity_sampling.py

Genetic-diversity phenotyping baseline.

This strategy selects candidates that are widely separated in marker
space using greedy maximin, also called farthest-point sampling.

The procedure is:

1. Standardize SNP-marker columns.
2. Choose an initial candidate.
3. Calculate each candidate's distance from the selected set.
4. Select the candidate with the largest minimum distance.
5. Repeat until the phenotyping budget is filled.

The returned values are unique, zero-based Python indices.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from src.baselines.base_strategy import BasePhenotypingStrategy


class DiversitySamplingStrategy(BasePhenotypingStrategy):
    """
    Select genetically diverse candidates using greedy maximin sampling.

    Parameters
    ----------
    initial_selection:
        Rule used to choose the first candidate.

        ``"random"``
            Select the first candidate randomly.

        ``"centroid_farthest"``
            Select the candidate farthest from the marker-space centroid.

    standardize_markers:
        If True, center and scale each polymorphic marker before calculating
        Euclidean distances.

    sort_indices:
        If True, sort the final selected indices. Sorting does not change
        which candidates were selected.
    """

    VALID_INITIAL_SELECTIONS = {
        "random",
        "centroid_farthest",
    }

    def __init__(
        self,
        initial_selection: str = "centroid_farthest",
        standardize_markers: bool = True,
        sort_indices: bool = True,
    ) -> None:
        super().__init__(name="diversity_sampling")

        if initial_selection not in self.VALID_INITIAL_SELECTIONS:
            raise ValueError(
                "'initial_selection' must be one of "
                f"{sorted(self.VALID_INITIAL_SELECTIONS)}."
            )

        if not isinstance(standardize_markers, bool):
            raise TypeError("'standardize_markers' must be Boolean.")

        if not isinstance(sort_indices, bool):
            raise TypeError("'sort_indices' must be Boolean.")

        self.initial_selection = initial_selection
        self.standardize_markers = standardize_markers
        self.sort_indices = sort_indices

    def select(
        self,
        candidate_data: Mapping[str, Any],
        number_to_phenotype: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Select a genetically diverse subset.

        ``candidate_data`` must contain a ``marker_matrix`` field with one
        row per candidate and one column per SNP marker.
        """
        population_size, number_to_phenotype = self.validate_inputs(
            candidate_data=candidate_data,
            number_to_phenotype=number_to_phenotype,
            rng=rng,
        )

        marker_matrix = self.require_field(
            candidate_data,
            "marker_matrix",
        )

        marker_matrix = self.validate_matrix(
            marker_matrix,
            field_name="marker_matrix",
            expected_rows=population_size,
        )

        if number_to_phenotype == population_size:
            selected_indices = np.arange(
                population_size,
                dtype=np.int64,
            )

            return self.validate_selection(
                selected_indices=selected_indices,
                population_size=population_size,
                number_to_phenotype=number_to_phenotype,
            )

        marker_matrix = self._prepare_marker_matrix(
            marker_matrix
        )

        if marker_matrix is None:
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

        first_index = self._select_initial_candidate(
            marker_matrix=marker_matrix,
            rng=rng,
        )

        selected_indices = self._greedy_maximin(
            marker_matrix=marker_matrix,
            first_index=first_index,
            number_to_select=number_to_phenotype,
        )

        if self.sort_indices:
            selected_indices = np.sort(selected_indices)

        return self.validate_selection(
            selected_indices=selected_indices,
            population_size=population_size,
            number_to_phenotype=number_to_phenotype,
        )

    def _prepare_marker_matrix(
        self,
        marker_matrix: np.ndarray,
    ) -> np.ndarray | None:
        """
        Remove noninformative markers and optionally standardize columns.
        """
        marker_matrix = np.asarray(
            marker_matrix,
            dtype=np.float64,
        )

        marker_variance = marker_matrix.var(axis=0)

        polymorphic_mask = (
            np.isfinite(marker_variance)
            & (marker_variance > 0.0)
        )

        if not np.any(polymorphic_mask):
            return None

        prepared_matrix = marker_matrix[
            :,
            polymorphic_mask,
        ].copy()

        if self.standardize_markers:
            marker_means = prepared_matrix.mean(
                axis=0
            )

            marker_standard_deviations = prepared_matrix.std(
                axis=0,
                ddof=0,
            )

            usable = (
                np.isfinite(marker_standard_deviations)
                & (marker_standard_deviations > 0.0)
            )

            if not np.any(usable):
                return None

            prepared_matrix = prepared_matrix[
                :,
                usable,
            ]
            marker_means = marker_means[usable]
            marker_standard_deviations = (
                marker_standard_deviations[usable]
            )

            prepared_matrix = (
                prepared_matrix - marker_means
            ) / marker_standard_deviations

        return prepared_matrix

    def _select_initial_candidate(
        self,
        marker_matrix: np.ndarray,
        rng: np.random.Generator,
    ) -> int:
        """Choose the first candidate in the greedy procedure."""
        population_size = marker_matrix.shape[0]

        if self.initial_selection == "random":
            return int(
                rng.integers(
                    low=0,
                    high=population_size,
                )
            )

        centroid = marker_matrix.mean(
            axis=0
        )

        squared_distance_from_centroid = np.sum(
            (marker_matrix - centroid) ** 2,
            axis=1,
        )

        return int(
            np.argmax(
                squared_distance_from_centroid
            )
        )

    @staticmethod
    def _greedy_maximin(
        marker_matrix: np.ndarray,
        first_index: int,
        number_to_select: int,
    ) -> np.ndarray:
        """
        Perform greedy farthest-point sampling.

        At every iteration, select the candidate whose distance from its
        nearest already selected candidate is greatest.
        """
        population_size = marker_matrix.shape[0]

        selected = np.empty(
            number_to_select,
            dtype=np.int64,
        )

        selected[0] = first_index

        is_selected = np.zeros(
            population_size,
            dtype=bool,
        )

        is_selected[first_index] = True

        squared_minimum_distances = np.sum(
            (
                marker_matrix
                - marker_matrix[first_index]
            )
            ** 2,
            axis=1,
        )

        squared_minimum_distances[first_index] = (
            -np.inf
        )

        for position in range(
            1,
            number_to_select,
        ):
            next_index = int(
                np.argmax(
                    squared_minimum_distances
                )
            )

            selected[position] = next_index
            is_selected[next_index] = True

            squared_distance_to_new_candidate = (
                np.sum(
                    (
                        marker_matrix
                        - marker_matrix[next_index]
                    )
                    ** 2,
                    axis=1,
                )
            )

            squared_minimum_distances = np.minimum(
                squared_minimum_distances,
                squared_distance_to_new_candidate,
            )

            squared_minimum_distances[
                is_selected
            ] = -np.inf

        return selected

    def __repr__(self) -> str:
        """Return a readable representation."""
        return (
            f"{self.__class__.__name__}("
            f"name={self.name!r}, "
            f"initial_selection={self.initial_selection!r}, "
            f"standardize_markers={self.standardize_markers!r}, "
            f"sort_indices={self.sort_indices!r})"
        )
