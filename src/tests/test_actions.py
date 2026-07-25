"""Unit tests for environment actions."""

import numpy as np

from src.environment.actions import (
    PhenotypingAction,
    action_name,
    build_action_mask,
    validate_action,
)


def main() -> None:
    assert validate_action(0) == PhenotypingAction.RANDOM
    assert validate_action(np.int64(4)) == PhenotypingAction.STOP
    assert action_name(2) == "Highest-PEV batch"

    initial_mask = build_action_mask(
        population_size=1000,
        number_phenotyped=0,
        batch_size=25,
        maximum_phenotypes=200,
        minimum_training_size=50,
        model_available=False,
        uncertainty_available=False,
    )

    assert np.array_equal(
        initial_mask,
        np.array([True, True, False, False, False]),
    )

    fitted_mask = build_action_mask(
        population_size=1000,
        number_phenotyped=50,
        batch_size=25,
        maximum_phenotypes=200,
        minimum_training_size=50,
        model_available=True,
        uncertainty_available=True,
    )

    assert fitted_mask.all()

    exhausted_mask = build_action_mask(
        population_size=1000,
        number_phenotyped=200,
        batch_size=25,
        maximum_phenotypes=200,
        minimum_training_size=50,
        model_available=True,
        uncertainty_available=True,
    )

    assert np.array_equal(
        exhausted_mask,
        np.array([False, False, False, False, True]),
    )

    try:
        validate_action(10)
    except ValueError:
        pass
    else:
        raise AssertionError("Unknown actions must fail.")

    print("All action checks passed.")


if __name__ == "__main__":
    main()
