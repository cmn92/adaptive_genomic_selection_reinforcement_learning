"""Unit tests for RL state construction."""

import numpy as np

from src.environment.state import (
    BreedingStateSnapshot,
    OBSERVATION_NAMES,
    build_observation,
    observation_size,
)


def main() -> None:
    snapshot = BreedingStateSnapshot(
        generation=2,
        maximum_generations=10,
        number_phenotyped=50,
        maximum_phenotypes=200,
        model_available=True,
        mean_pev=1.2,
        max_pev=2.5,
        mean_reliability=0.4,
        mean_predicted_gebv=1.1,
        predicted_gebv_standard_deviation=0.7,
        candidate_marker_diversity=25.0,
        training_marker_diversity=20.0,
        prediction_residual_mae=0.4,
        prediction_residual_rmse=0.6,
        prediction_residual_bias=-0.2,
        previous_genetic_gain=0.8,
        previous_variance_retention=0.75,
    )

    observation = build_observation(snapshot)

    assert observation.shape == (observation_size(),)
    assert observation_size() == len(OBSERVATION_NAMES)
    assert observation.dtype == np.float32
    assert np.isfinite(observation).all()
    assert np.all(observation >= -1.0)
    assert np.all(observation <= 1.0)

    assert observation[
        OBSERVATION_NAMES.index("model_available")
    ] == 1.0

    no_model = build_observation(
        BreedingStateSnapshot(
            generation=1,
            maximum_generations=10,
            number_phenotyped=0,
            maximum_phenotypes=200,
            model_available=False,
        )
    )

    assert no_model[
        OBSERVATION_NAMES.index("model_available")
    ] == -1.0
    assert no_model[
        OBSERVATION_NAMES.index("phenotyping_fraction")
    ] == -1.0
    assert no_model[
        OBSERVATION_NAMES.index("remaining_budget_fraction")
    ] == 1.0

    print("All state checks passed.")


if __name__ == "__main__":
    main()
