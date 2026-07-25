"""
discretizer.py

Convert the continuous breeding-environment observation into a
discrete state that can be used by tabular Q-learning.

The discretizer assigns each observation feature to one of several bins.
The complete discrete state is represented as a tuple of bin indices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from src.environment.state import (
    observation_size as breeding_observation_size,
)


@dataclass(frozen=True)
class ObservationDiscretizer:
    """
    Discretize normalized observations in the interval [-1, 1].

    Parameters
    ----------
    bins_per_feature:
        Either one integer applied to every observation feature or one
        integer per feature.

    observation_size:
        Number of values in the observation vector.
    """

    bins_per_feature: int | tuple[int, ...] = 5
    observation_size: int = breeding_observation_size()

    def __post_init__(self) -> None:
        if (
            isinstance(self.observation_size, bool)
            or not isinstance(self.observation_size, (int, np.integer))
            or self.observation_size < 1
        ):
            raise ValueError(
                "'observation_size' must be a positive integer."
            )

        bins = self._normalize_bins(
            self.bins_per_feature,
            self.observation_size,
        )

        object.__setattr__(
            self,
            "_bins",
            bins,
        )

        edges = tuple(
            np.linspace(
                -1.0,
                1.0,
                num=bin_count + 1,
                dtype=np.float64,
            )[1:-1]
            for bin_count in bins
        )

        object.__setattr__(
            self,
            "_edges",
            edges,
        )

    @staticmethod
    def _normalize_bins(
        bins_per_feature: int | tuple[int, ...],
        observation_size: int,
    ) -> tuple[int, ...]:
        """Validate and standardize bin counts."""
        if isinstance(bins_per_feature, (int, np.integer)):
            if isinstance(bins_per_feature, bool):
                raise TypeError(
                    "'bins_per_feature' must not be Boolean."
                )

            if bins_per_feature < 2:
                raise ValueError(
                    "Each feature must use at least two bins."
                )

            return tuple(
                int(bins_per_feature)
                for _ in range(observation_size)
            )

        if not isinstance(bins_per_feature, tuple):
            raise TypeError(
                "'bins_per_feature' must be an integer or tuple."
            )

        if len(bins_per_feature) != observation_size:
            raise ValueError(
                "The number of per-feature bin counts must match "
                "'observation_size'."
            )

        normalized: list[int] = []

        for index, value in enumerate(bins_per_feature):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or value < 2
            ):
                raise ValueError(
                    f"Bin count at position {index} must be an integer "
                    "of at least two."
                )

            normalized.append(int(value))

        return tuple(normalized)

    @property
    def bins(self) -> tuple[int, ...]:
        """Return the number of bins for each feature."""
        return self._bins

    @property
    def number_of_states(self) -> int:
        """Return the theoretical number of discrete states."""
        return int(np.prod(self._bins, dtype=np.int64))

    def transform(
        self,
        observation: np.ndarray | Iterable[float],
    ) -> tuple[int, ...]:
        """
        Convert one normalized observation into a tuple of bin indices.
        """
        values = np.asarray(
            observation,
            dtype=np.float64,
        )

        if values.shape != (self.observation_size,):
            raise ValueError(
                "'observation' must have shape "
                f"({self.observation_size},)."
            )

        if not np.isfinite(values).all():
            raise ValueError(
                "'observation' cannot contain NaN or infinity."
            )

        values = np.clip(
            values,
            -1.0,
            1.0,
        )

        discrete_state = tuple(
            int(
                np.digitize(
                    values[index],
                    self._edges[index],
                    right=False,
                )
            )
            for index in range(self.observation_size)
        )

        return discrete_state

    def transform_batch(
        self,
        observations: np.ndarray,
    ) -> list[tuple[int, ...]]:
        """Discretize a two-dimensional batch of observations."""
        values = np.asarray(
            observations,
            dtype=np.float64,
        )

        if values.ndim != 2:
            raise ValueError(
                "'observations' must be a two-dimensional array."
            )

        if values.shape[1] != self.observation_size:
            raise ValueError(
                "Observation width does not match the configured size."
            )

        return [
            self.transform(row)
            for row in values
        ]
