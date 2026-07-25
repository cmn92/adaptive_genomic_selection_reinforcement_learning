"""
actions.py

Discrete actions available to the reinforcement-learning phenotyping agent.

Each non-stop action selects one batch of currently unphenotyped candidates.
The environment then phenotypes that batch and updates the genomic prediction
model when enough training records are available.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any

import numpy as np


class PhenotypingAction(IntEnum):
    """Discrete actions used by the breeding environment."""

    RANDOM = 0
    DIVERSITY = 1
    HIGHEST_PEV = 2
    HIGHEST_GEBV = 3
    STOP = 4


ACTION_LABELS = {
    PhenotypingAction.RANDOM: "Random batch",
    PhenotypingAction.DIVERSITY: "Diversity batch",
    PhenotypingAction.HIGHEST_PEV: "Highest-PEV batch",
    PhenotypingAction.HIGHEST_GEBV: "Highest-GEBV batch",
    PhenotypingAction.STOP: "Stop and finalize generation",
}


def validate_action(action: Any) -> PhenotypingAction:
    """
    Validate and convert an action to PhenotypingAction.

    Gymnasium and NumPy may supply Python integers or NumPy integer scalars.
    Boolean values are rejected even though bool subclasses int.
    """
    if isinstance(action, (bool, np.bool_)):
        raise TypeError("'action' must be an integer action code.")

    if not isinstance(action, (int, np.integer, PhenotypingAction)):
        raise TypeError("'action' must be an integer action code.")

    try:
        return PhenotypingAction(int(action))
    except ValueError as exc:
        valid_codes = [int(item) for item in PhenotypingAction]
        raise ValueError(
            f"Unknown action {action!r}. Valid action codes are {valid_codes}."
        ) from exc


def action_name(action: Any) -> str:
    """Return the readable name of an action."""
    validated = validate_action(action)
    return ACTION_LABELS[validated]


def build_action_mask(
    *,
    population_size: int,
    number_phenotyped: int,
    batch_size: int,
    maximum_phenotypes: int,
    minimum_training_size: int,
    model_available: bool,
    uncertainty_available: bool,
) -> np.ndarray:
    """
    Build a Boolean mask for the five discrete actions.

    Rules
    -----
    RANDOM and DIVERSITY are valid while at least one complete batch remains.

    HIGHEST_PEV requires:
    - a fitted model;
    - current uncertainty estimates;
    - a complete batch remaining.

    HIGHEST_GEBV requires:
    - a fitted model;
    - a complete batch remaining.

    STOP requires at least minimum_training_size phenotyped candidates.
    """
    integer_values = {
        "population_size": population_size,
        "number_phenotyped": number_phenotyped,
        "batch_size": batch_size,
        "maximum_phenotypes": maximum_phenotypes,
        "minimum_training_size": minimum_training_size,
    }

    for name, value in integer_values.items():
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise TypeError(f"'{name}' must be an integer.")

    if population_size < 1:
        raise ValueError("'population_size' must be positive.")
    if number_phenotyped < 0:
        raise ValueError("'number_phenotyped' cannot be negative.")
    if batch_size < 1:
        raise ValueError("'batch_size' must be positive.")
    if maximum_phenotypes < 1:
        raise ValueError("'maximum_phenotypes' must be positive.")
    if minimum_training_size < 1:
        raise ValueError("'minimum_training_size' must be positive.")

    if maximum_phenotypes > population_size:
        raise ValueError(
            "'maximum_phenotypes' cannot exceed the population size."
        )

    remaining_budget = maximum_phenotypes - number_phenotyped
    remaining_candidates = population_size - number_phenotyped

    complete_batch_available = (
        remaining_budget >= batch_size
        and remaining_candidates >= batch_size
    )

    mask = np.zeros(len(PhenotypingAction), dtype=bool)

    mask[PhenotypingAction.RANDOM] = complete_batch_available
    mask[PhenotypingAction.DIVERSITY] = complete_batch_available

    mask[PhenotypingAction.HIGHEST_PEV] = (
        complete_batch_available
        and bool(model_available)
        and bool(uncertainty_available)
    )

    mask[PhenotypingAction.HIGHEST_GEBV] = (
        complete_batch_available
        and bool(model_available)
    )

    mask[PhenotypingAction.STOP] = (
        number_phenotyped >= minimum_training_size
    )

    return mask
