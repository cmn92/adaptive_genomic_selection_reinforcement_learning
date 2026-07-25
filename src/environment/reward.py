"""
reward.py

Reward calculation for adaptive phenotyping.

Intermediate batch rewards penalize phenotyping cost. Finalization adds reward
for realized genetic gain and retained genetic variance.

True genetic values are used only inside the hidden simulator reward. They are
not exposed in the observation supplied to the agent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RewardConfig:
    """Weights and scaling constants for the RL reward."""

    genetic_gain_weight: float = 1.0
    variance_retention_weight: float = 0.5
    phenotyping_cost_weight: float = 0.2
    invalid_action_penalty: float = 1.0
    gain_scale: float = 1.0

    def __post_init__(self) -> None:
        values = {
            "genetic_gain_weight": self.genetic_gain_weight,
            "variance_retention_weight": (
                self.variance_retention_weight
            ),
            "phenotyping_cost_weight": (
                self.phenotyping_cost_weight
            ),
            "invalid_action_penalty": (
                self.invalid_action_penalty
            ),
            "gain_scale": self.gain_scale,
        }

        for name, value in values.items():
            numeric = float(value)
            if not np.isfinite(numeric):
                raise ValueError(f"'{name}' must be finite.")
            if numeric < 0:
                raise ValueError(f"'{name}' cannot be negative.")

        if self.gain_scale <= 0:
            raise ValueError("'gain_scale' must be strictly positive.")


@dataclass(frozen=True)
class RewardBreakdown:
    """Components of one environment reward."""

    total: float
    genetic_gain_component: float
    variance_component: float
    cost_component: float
    invalid_action_component: float


def batch_cost_reward(
    *,
    batch_size: int,
    maximum_phenotypes: int,
    config: RewardConfig,
) -> RewardBreakdown:
    """Return the immediate cost penalty for one phenotyping batch."""
    if not isinstance(config, RewardConfig):
        raise TypeError("'config' must be a RewardConfig instance.")

    for name, value in {
        "batch_size": batch_size,
        "maximum_phenotypes": maximum_phenotypes,
    }.items():
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise TypeError(f"'{name}' must be an integer.")

    if batch_size < 1:
        raise ValueError("'batch_size' must be positive.")
    if maximum_phenotypes < 1:
        raise ValueError("'maximum_phenotypes' must be positive.")
    if batch_size > maximum_phenotypes:
        raise ValueError(
            "'batch_size' cannot exceed 'maximum_phenotypes'."
        )

    cost = -config.phenotyping_cost_weight * (
        batch_size / maximum_phenotypes
    )

    return RewardBreakdown(
        total=float(cost),
        genetic_gain_component=0.0,
        variance_component=0.0,
        cost_component=float(cost),
        invalid_action_component=0.0,
    )


def final_generation_reward(
    *,
    realized_genetic_gain: float,
    variance_retention: float,
    number_phenotyped: int,
    maximum_phenotypes: int,
    config: RewardConfig,
) -> RewardBreakdown:
    """
    Calculate the complete reward when a breeding generation is finalized.

    The phenotyping cost is based on total phenotypes used in the generation.
    Therefore, environments should choose either:

    - intermediate batch cost plus a terminal reward with zero cost weight; or
    - no intermediate cost plus this complete terminal reward.

    BreedingEnv uses intermediate batch penalties and a terminal reward whose
    cost component covers only any difference not already charged.
    """
    if not isinstance(config, RewardConfig):
        raise TypeError("'config' must be a RewardConfig instance.")

    gain = float(realized_genetic_gain)
    retained = float(variance_retention)

    if not np.isfinite(gain):
        raise ValueError("'realized_genetic_gain' must be finite.")
    if not np.isfinite(retained):
        raise ValueError("'variance_retention' must be finite.")
    if retained < 0:
        raise ValueError("'variance_retention' cannot be negative.")

    for name, value in {
        "number_phenotyped": number_phenotyped,
        "maximum_phenotypes": maximum_phenotypes,
    }.items():
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise TypeError(f"'{name}' must be an integer.")

    if not 0 <= number_phenotyped <= maximum_phenotypes:
        raise ValueError(
            "'number_phenotyped' must be between zero and the maximum."
        )

    gain_component = (
        config.genetic_gain_weight
        * np.tanh(gain / config.gain_scale)
    )

    # Retaining all initial variance gives the full positive variance reward.
    # Retaining none gives zero. Values above one are clipped.
    variance_component = (
        config.variance_retention_weight
        * np.clip(retained, 0.0, 1.0)
    )

    cost_component = -config.phenotyping_cost_weight * (
        number_phenotyped / maximum_phenotypes
    )

    total = (
        gain_component
        + variance_component
        + cost_component
    )

    return RewardBreakdown(
        total=float(total),
        genetic_gain_component=float(gain_component),
        variance_component=float(variance_component),
        cost_component=float(cost_component),
        invalid_action_component=0.0,
    )


def invalid_action_reward(
    config: RewardConfig,
) -> RewardBreakdown:
    """Return the configured penalty for an invalid action."""
    if not isinstance(config, RewardConfig):
        raise TypeError("'config' must be a RewardConfig instance.")

    penalty = -float(config.invalid_action_penalty)

    return RewardBreakdown(
        total=penalty,
        genetic_gain_component=0.0,
        variance_component=0.0,
        cost_component=0.0,
        invalid_action_component=penalty,
    )
