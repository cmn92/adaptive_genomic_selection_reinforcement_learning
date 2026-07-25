"""
state.py

Observation construction for the reinforcement-learning breeding environment.

The observation contains only information that could be available to a breeder
during the simulated program. True breeding values and realized genetic gain
from the current unfinished generation are not included.

All returned observation values are finite float32 values bounded by [-1, 1].
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


OBSERVATION_NAMES = (
    "generation_progress",
    "generation_remaining_fraction",
    "phenotyping_fraction",
    "remaining_budget_fraction",
    "budget_pressure",
    "model_available",
    "mean_log_pev",
    "max_log_pev",
    "mean_reliability",
    "mean_predicted_gebv",
    "predicted_gebv_spread",
    "candidate_marker_diversity",
    "training_marker_diversity",
    "training_diversity_fraction",
    "prediction_residual_mae",
    "prediction_residual_rmse",
    "prediction_residual_bias",
    "previous_genetic_gain",
    "previous_variance_retention",
)


@dataclass(frozen=True)
class BreedingStateSnapshot:
    """Raw values used to construct one RL observation."""

    generation: int
    maximum_generations: int
    number_phenotyped: int
    maximum_phenotypes: int
    model_available: bool
    mean_pev: float = 0.0
    max_pev: float = 0.0
    mean_reliability: float = 0.0
    mean_predicted_gebv: float = 0.0
    predicted_gebv_standard_deviation: float = 0.0
    candidate_marker_diversity: float = 0.0
    training_marker_diversity: float = 0.0
    prediction_residual_mae: float = 0.0
    prediction_residual_rmse: float = 0.0
    prediction_residual_bias: float = 0.0
    previous_genetic_gain: float = 0.0
    previous_variance_retention: float = 1.0


def observation_size() -> int:
    """Return the fixed observation-vector length."""
    return len(OBSERVATION_NAMES)


def _finite_float(value: float, name: str) -> float:
    """Convert a value to a finite float."""
    numeric = float(value)
    if not np.isfinite(numeric):
        raise ValueError(f"'{name}' must be finite.")
    return numeric


def build_observation(
    snapshot: BreedingStateSnapshot,
) -> np.ndarray:
    """
    Convert a raw state snapshot into a normalized observation vector.

    Scaling choices
    ---------------
    Fractions are mapped from [0, 1] to [-1, 1].

    PEV is transformed with log1p and then squashed with tanh because its
    absolute scale depends on the genomic relationship model.

    GEBV summaries, marker diversity, prediction residuals, and prior gain
    are squashed with tanh to avoid assuming a fixed biological scale.

    Variance retention is centered around 1.0, where zero in the transformed
    observation means all initial variance was retained.
    """
    if not isinstance(snapshot, BreedingStateSnapshot):
        raise TypeError(
            "'snapshot' must be a BreedingStateSnapshot instance."
        )

    integer_fields = {
        "generation": snapshot.generation,
        "maximum_generations": snapshot.maximum_generations,
        "number_phenotyped": snapshot.number_phenotyped,
        "maximum_phenotypes": snapshot.maximum_phenotypes,
    }

    for name, value in integer_fields.items():
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise TypeError(f"'{name}' must be an integer.")

    if snapshot.maximum_generations < 1:
        raise ValueError("'maximum_generations' must be positive.")
    if not 1 <= snapshot.generation <= snapshot.maximum_generations:
        raise ValueError(
            "'generation' must be within the configured episode horizon."
        )
    if snapshot.maximum_phenotypes < 1:
        raise ValueError("'maximum_phenotypes' must be positive.")
    if not 0 <= snapshot.number_phenotyped <= snapshot.maximum_phenotypes:
        raise ValueError(
            "'number_phenotyped' must lie between zero and the maximum."
        )

    mean_pev = max(
        _finite_float(snapshot.mean_pev, "mean_pev"),
        0.0,
    )
    max_pev = max(
        _finite_float(snapshot.max_pev, "max_pev"),
        0.0,
    )
    mean_reliability = np.clip(
        _finite_float(
            snapshot.mean_reliability,
            "mean_reliability",
        ),
        0.0,
        1.0,
    )

    mean_gebv = _finite_float(
        snapshot.mean_predicted_gebv,
        "mean_predicted_gebv",
    )
    gebv_spread = max(
        _finite_float(
            snapshot.predicted_gebv_standard_deviation,
            "predicted_gebv_standard_deviation",
        ),
        0.0,
    )
    candidate_marker_diversity = max(
        _finite_float(
            snapshot.candidate_marker_diversity,
            "candidate_marker_diversity",
        ),
        0.0,
    )
    marker_diversity = max(
        _finite_float(
            snapshot.training_marker_diversity,
            "training_marker_diversity",
        ),
        0.0,
    )
    residual_mae = max(
        _finite_float(
            snapshot.prediction_residual_mae,
            "prediction_residual_mae",
        ),
        0.0,
    )
    residual_rmse = max(
        _finite_float(
            snapshot.prediction_residual_rmse,
            "prediction_residual_rmse",
        ),
        0.0,
    )
    residual_bias = _finite_float(
        snapshot.prediction_residual_bias,
        "prediction_residual_bias",
    )
    previous_gain = _finite_float(
        snapshot.previous_genetic_gain,
        "previous_genetic_gain",
    )
    variance_retention = max(
        _finite_float(
            snapshot.previous_variance_retention,
            "previous_variance_retention",
        ),
        0.0,
    )

    generation_fraction = (
        snapshot.generation / snapshot.maximum_generations
    )
    phenotyping_fraction = (
        snapshot.number_phenotyped / snapshot.maximum_phenotypes
    )
    remaining_fraction = 1.0 - phenotyping_fraction
    remaining_generation_fraction = (
        (
            snapshot.maximum_generations
            - snapshot.generation
        )
        / snapshot.maximum_generations
    )
    budget_pressure = (
        phenotyping_fraction / generation_fraction
        if generation_fraction > 0
        else 0.0
    )
    training_diversity_fraction = (
        marker_diversity / candidate_marker_diversity
        if candidate_marker_diversity > 0
        else 0.0
    )

    observation = np.array(
        [
            2.0 * generation_fraction - 1.0,
            2.0 * remaining_generation_fraction - 1.0,
            2.0 * phenotyping_fraction - 1.0,
            2.0 * remaining_fraction - 1.0,
            2.0 * np.clip(budget_pressure, 0.0, 1.0) - 1.0,
            1.0 if snapshot.model_available else -1.0,
            np.tanh(np.log1p(mean_pev)),
            np.tanh(np.log1p(max_pev)),
            2.0 * mean_reliability - 1.0,
            np.tanh(mean_gebv / 5.0),
            np.tanh(gebv_spread / 5.0),
            np.tanh(candidate_marker_diversity / 10.0),
            np.tanh(marker_diversity / 10.0),
            2.0 * np.clip(
                training_diversity_fraction,
                0.0,
                1.0,
            )
            - 1.0,
            np.tanh(residual_mae / 5.0),
            np.tanh(residual_rmse / 5.0),
            np.tanh(residual_bias / 5.0),
            np.tanh(previous_gain / 5.0),
            np.tanh(variance_retention - 1.0),
        ],
        dtype=np.float32,
    )

    if observation.shape != (observation_size(),):
        raise RuntimeError("Observation vector has an unexpected shape.")

    if not np.isfinite(observation).all():
        raise RuntimeError("Observation contains nonfinite values.")

    return np.clip(observation, -1.0, 1.0)
