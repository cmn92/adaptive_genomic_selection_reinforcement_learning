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
    OBSERVATION_NAMES,
    observation_size as breeding_observation_size,
)


COMPACT_BREEDING_FEATURES = (
    "generation_progress",
    "phenotyping_fraction",
    "model_available",
    "mean_reliability",
    "previous_variance_retention",
)

COMPACT_BREEDING_BINS = (
    4,
    5,
    2,
    4,
    3,
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

    feature_indices:
        Optional zero-based observation columns to discretize. When supplied,
        the full observation is still validated against ``observation_size``,
        but the returned discrete state contains only these selected features.
    """

    bins_per_feature: int | tuple[int, ...] = 5
    observation_size: int = breeding_observation_size()
    feature_indices: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.observation_size, bool)
            or not isinstance(self.observation_size, (int, np.integer))
            or self.observation_size < 1
        ):
            raise ValueError(
                "'observation_size' must be a positive integer."
            )

        feature_indices = self._normalize_feature_indices(
            self.feature_indices,
            self.observation_size,
        )

        bins = self._normalize_bins(
            self.bins_per_feature,
            len(feature_indices),
        )

        object.__setattr__(
            self,
            "_feature_indices",
            feature_indices,
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
    def _normalize_feature_indices(
        feature_indices: tuple[int, ...] | None,
        observation_size: int,
    ) -> tuple[int, ...]:
        """Validate and standardize selected observation columns."""
        if feature_indices is None:
            return tuple(range(observation_size))

        if not isinstance(feature_indices, tuple):
            raise TypeError(
                "'feature_indices' must be a tuple or None."
            )

        if len(feature_indices) < 1:
            raise ValueError(
                "'feature_indices' must contain at least one index."
            )

        normalized: list[int] = []

        for position, value in enumerate(feature_indices):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
            ):
                raise TypeError(
                    f"Feature index at position {position} must be an integer."
                )

            index = int(value)

            if not 0 <= index < observation_size:
                raise ValueError(
                    f"Feature index {index} is outside the observation width."
                )

            normalized.append(index)

        if len(set(normalized)) != len(normalized):
            raise ValueError(
                "'feature_indices' cannot contain duplicates."
            )

        return tuple(normalized)

    @staticmethod
    def _normalize_bins(
        bins_per_feature: int | tuple[int, ...],
        number_of_features: int,
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
                for _ in range(number_of_features)
            )

        if not isinstance(bins_per_feature, tuple):
            raise TypeError(
                "'bins_per_feature' must be an integer or tuple."
            )

        if len(bins_per_feature) != number_of_features:
            raise ValueError(
                "The number of per-feature bin counts must match "
                "the number of discretized features."
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
    def feature_indices_(self) -> tuple[int, ...]:
        """Return the selected full-observation feature indices."""
        return self._feature_indices

    @property
    def state_size(self) -> int:
        """Return the number of values in the discrete state tuple."""
        return len(self._feature_indices)

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
        selected_values = values[
            list(self._feature_indices)
        ]

        discrete_state = tuple(
            int(
                np.digitize(
                    selected_values[index],
                    self._edges[index],
                    right=False,
                )
            )
            for index in range(self.state_size)
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


def feature_indices_for_names(
    feature_names: tuple[str, ...],
) -> tuple[int, ...]:
    """Return observation indices for named breeding-state features."""
    indices: list[int] = []

    for name in feature_names:
        if name not in OBSERVATION_NAMES:
            raise ValueError(
                f"Unknown observation feature: {name!r}."
            )

        indices.append(
            OBSERVATION_NAMES.index(name)
        )

    return tuple(indices)


def compact_breeding_discretizer() -> ObservationDiscretizer:
    """Return the compact recurring state space for tabular Q-learning."""
    return ObservationDiscretizer(
        bins_per_feature=COMPACT_BREEDING_BINS,
        observation_size=breeding_observation_size(),
        feature_indices=feature_indices_for_names(
            COMPACT_BREEDING_FEATURES
        ),
    )


def compact_breeding_feature_indices() -> tuple[int, ...]:
    """Return compact recurring observation columns for function approximators."""
    return feature_indices_for_names(COMPACT_BREEDING_FEATURES)
