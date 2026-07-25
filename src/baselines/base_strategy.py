"""
base_strategy.py

Shared interface for all phenotyping-selection strategies.

Each strategy receives information about the current candidate population
and returns zero-based Python indices identifying which candidates should
be phenotyped.

Examples of later strategies:
- random sampling
- fixed sampling
- diversity-based sampling
- active learning
- reinforcement learning
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping

import numpy as np


class StrategyValidationError(ValueError):
    """Raised when a phenotyping strategy returns an invalid selection."""


class BasePhenotypingStrategy(ABC):
    """
    Abstract base class for phenotyping-selection strategies.

    Every concrete strategy must implement the select() method.

    The strategy should return a one-dimensional NumPy array containing
    unique, zero-based candidate indices.
    """

    def __init__(self, name: str) -> None:
        """
        Initialize the strategy.

        Parameters
        ----------
        name:
            Human-readable strategy name used in experiment outputs.
        """
        if not isinstance(name, str) or not name.strip():
            raise ValueError("'name' must be a non-empty string.")

        self.name = name.strip()

    @abstractmethod
    def select(
        self,
        candidate_data: Mapping[str, Any],
        number_to_phenotype: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Select candidates for phenotyping.

        Parameters
        ----------
        candidate_data:
            Information about the current candidate population.

            Expected basic fields include:

            {
                "generation": int,
                "population_size": int,
                "individual_ids": list[str]
            }

            Some strategies may require additional fields, such as:

            {
                "marker_matrix": numpy.ndarray,
                "predicted_gebv": numpy.ndarray,
                "prediction_uncertainty": numpy.ndarray
            }

        number_to_phenotype:
            Number of candidates that must be selected.

        rng:
            NumPy random-number generator supplied by the experiment
            runner for reproducibility.

        Returns
        -------
        numpy.ndarray
            One-dimensional array of unique zero-based candidate indices.
        """
        raise NotImplementedError

    def validate_inputs(
        self,
        candidate_data: Mapping[str, Any],
        number_to_phenotype: int,
        rng: np.random.Generator,
    ) -> tuple[int, int]:
        """
        Validate the common inputs supplied to every strategy.

        Returns
        -------
        tuple[int, int]
            Population size and validated number to phenotype.
        """
        if not isinstance(candidate_data, Mapping):
            raise TypeError(
                "'candidate_data' must be a mapping such as a dictionary."
            )

        if "population_size" not in candidate_data:
            raise KeyError(
                "'candidate_data' must contain a 'population_size' field."
            )

        population_size = self._validate_positive_integer(
            candidate_data["population_size"],
            "candidate_data['population_size']",
        )

        number_to_phenotype = self._validate_positive_integer(
            number_to_phenotype,
            "number_to_phenotype",
        )

        if number_to_phenotype > population_size:
            raise ValueError(
                "'number_to_phenotype' cannot exceed the current "
                f"population size of {population_size}."
            )

        if not isinstance(rng, np.random.Generator):
            raise TypeError(
                "'rng' must be an instance of numpy.random.Generator."
            )

        individual_ids = candidate_data.get("individual_ids")

        if individual_ids is not None:
            if len(individual_ids) != population_size:
                raise ValueError(
                    "The number of individual IDs does not match "
                    "'population_size'."
                )

        return population_size, number_to_phenotype

    def validate_selection(
        self,
        selected_indices: np.ndarray | list[int],
        population_size: int,
        number_to_phenotype: int,
    ) -> np.ndarray:
        """
        Validate and standardize indices returned by a strategy.

        Parameters
        ----------
        selected_indices:
            Candidate indices produced by the strategy.

        population_size:
            Number of candidates in the current population.

        number_to_phenotype:
            Required number of selected candidates.

        Returns
        -------
        numpy.ndarray
            Validated array with dtype int64.
        """
        indices = np.asarray(selected_indices)

        if indices.ndim != 1:
            raise StrategyValidationError(
                "A strategy must return a one-dimensional array of indices."
            )

        if indices.size != number_to_phenotype:
            raise StrategyValidationError(
                f"Strategy '{self.name}' returned {indices.size} candidates, "
                f"but {number_to_phenotype} were required."
            )

        if not np.issubdtype(indices.dtype, np.number):
            raise StrategyValidationError(
                "Selected indices must be numeric."
            )

        numeric_indices = indices.astype(float)

        if np.any(~np.isfinite(numeric_indices)):
            raise StrategyValidationError(
                "Selected indices cannot contain NaN or infinite values."
            )

        if np.any(numeric_indices % 1 != 0):
            raise StrategyValidationError(
                "Selected indices must contain whole numbers."
            )

        indices = indices.astype(np.int64)

        if np.any(indices < 0):
            raise StrategyValidationError(
                "Python candidate indices must be zero-based and non-negative."
            )

        if np.any(indices >= population_size):
            raise StrategyValidationError(
                "At least one selected index exceeds the current "
                f"population size of {population_size}."
            )

        if np.unique(indices).size != indices.size:
            raise StrategyValidationError(
                "A strategy cannot select the same candidate more than once."
            )

        return indices

    @staticmethod
    def require_field(
        candidate_data: Mapping[str, Any],
        field_name: str,
    ) -> Any:
        """
        Retrieve a required strategy-specific field.

        This provides a clearer error when, for example, a diversity
        strategy is called without a marker matrix.
        """
        if field_name not in candidate_data:
            raise KeyError(
                f"'candidate_data' is missing the required field "
                f"'{field_name}'."
            )

        return candidate_data[field_name]

    @staticmethod
    def validate_vector(
        values: Any,
        *,
        field_name: str,
        expected_length: int,
        allow_missing: bool = False,
    ) -> np.ndarray:
        """
        Validate a one-dimensional candidate-level numeric vector.

        This will later be useful for GEBVs and prediction uncertainty.
        """
        vector = np.asarray(values, dtype=np.float64)

        if vector.ndim != 1:
            raise ValueError(
                f"'{field_name}' must be a one-dimensional vector."
            )

        if vector.size != expected_length:
            raise ValueError(
                f"'{field_name}' contains {vector.size} values, but "
                f"{expected_length} were expected."
            )

        if not allow_missing and np.any(~np.isfinite(vector)):
            raise ValueError(
                f"'{field_name}' cannot contain NaN or infinite values."
            )

        return vector

    @staticmethod
    def validate_matrix(
        values: Any,
        *,
        field_name: str,
        expected_rows: int,
    ) -> np.ndarray:
        """
        Validate a candidate-by-feature numeric matrix.

        This will later be used for SNP marker matrices and embeddings.
        """
        matrix = np.asarray(values, dtype=np.float64)

        if matrix.ndim != 2:
            raise ValueError(
                f"'{field_name}' must be a two-dimensional matrix."
            )

        if matrix.shape[0] != expected_rows:
            raise ValueError(
                f"'{field_name}' contains {matrix.shape[0]} rows, but "
                f"{expected_rows} were expected."
            )

        if np.any(~np.isfinite(matrix)):
            raise ValueError(
                f"'{field_name}' cannot contain NaN or infinite values."
            )

        return matrix

    @staticmethod
    def _validate_positive_integer(value: Any, name: str) -> int:
        """Validate a scalar positive integer."""
        if isinstance(value, (bool, np.bool_)):
            raise TypeError(f"'{name}' must be a positive integer.")

        if not np.isscalar(value):
            raise TypeError(f"'{name}' must be a scalar positive integer.")

        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"'{name}' must be a positive integer."
            ) from exc

        if not np.isfinite(numeric_value):
            raise ValueError(f"'{name}' must be finite.")

        if numeric_value % 1 != 0 or numeric_value < 1:
            raise ValueError(f"'{name}' must be a positive integer.")

        return int(numeric_value)

    def __repr__(self) -> str:
        """Return a readable representation of the strategy."""
        return f"{self.__class__.__name__}(name={self.name!r})"