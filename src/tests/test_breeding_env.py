"""
Integration test for the Gymnasium breeding environment.

The test uses one breeding generation, two batches of 25 candidates, and
automatic finalization at a 50-candidate budget.
"""

from pathlib import Path

import numpy as np

from src.environment.actions import PhenotypingAction
from src.environment.breeding_env import (
    BreedingEnv,
    BreedingEnvConfig,
)
from src.environment.r_bridge import RBreedingBridge
from src.environment.r_bridge import RBridgeError
from src.environment.reward import RewardConfig


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class _BridgeStub:
    """Minimal bridge stub for private batch-selection checks."""

    def get_phenotyped_indices(self) -> np.ndarray:
        return np.array([], dtype=np.int64)


class _UncertaintyFailureBridgeStub:
    """Minimal bridge stub for uncertainty-refresh fallback checks."""

    def fit_current_model(self, *, trait: int = 1) -> dict:
        return {
            "prediction_table": "predictions-available",
        }

    def compute_current_uncertainty(self, **kwargs) -> dict:
        raise RBridgeError(
            "synthetic uncertainty failure"
        )


def check_monomorphic_diversity_fallback() -> None:
    """DIVERSITY falls back to a valid random batch if markers are fixed."""
    env = BreedingEnv.__new__(BreedingEnv)
    env.config = BreedingEnvConfig(
        maximum_generations=1,
        batch_size=3,
        minimum_training_size=3,
        maximum_phenotypes=6,
    )
    env.bridge = _BridgeStub()
    env._rng = np.random.default_rng(12345)
    env._marker_matrix = np.zeros(
        (6, 4),
        dtype=np.float64,
    )

    selected = env._select_diverse_batch(
        np.arange(6, dtype=np.int64)
    )

    assert selected.shape == (3,)
    assert np.unique(selected).size == 3
    assert selected.min() >= 0
    assert selected.max() < 6
    assert np.all(np.diff(selected) >= 0)


def check_uncertainty_failure_keeps_model() -> None:
    """Failed PEV refresh disables uncertainty without losing predictions."""
    env = BreedingEnv.__new__(BreedingEnv)
    env.config = BreedingEnvConfig()
    env.bridge = _UncertaintyFailureBridgeStub()
    env._marker_matrix = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=np.float64,
    )
    env._prediction_table = None
    env._uncertainty_table = "stale-uncertainty"
    env._last_uncertainty_error = None

    env._refresh_model_and_uncertainty()

    assert env._prediction_table == "predictions-available"
    assert env._uncertainty_table is None
    assert "synthetic uncertainty failure" in (
        env._last_uncertainty_error
    )


def main() -> None:
    check_monomorphic_diversity_fallback()
    check_uncertainty_failure_keeps_model()

    bridge = RBreedingBridge(
        project_root=PROJECT_ROOT,
        population_file=(
            "data/initial_candidate_population.RData"
        ),
        seed=12345,
    )

    env = BreedingEnv(
        bridge=bridge,
        config=BreedingEnvConfig(
            maximum_generations=1,
            batch_size=25,
            minimum_training_size=50,
            maximum_phenotypes=50,
            number_of_parents=20,
            number_of_crosses=100,
            f1_per_cross=1,
            dh_per_f1=10,
            reps=1,
            trait=1,
            snp_chip=1,
            n_cores=1,
            seed=12345,
        ),
        reward_config=RewardConfig(),
    )

    observation, info = env.reset(seed=12345)

    assert env.observation_space.contains(observation)
    assert info["event"] == "reset"
    assert np.array_equal(
        info["action_mask"],
        np.array([True, True, False, False, False]),
    )

    observation, reward_1, terminated, truncated, info = env.step(
        PhenotypingAction.DIVERSITY
    )

    assert not terminated
    assert not truncated
    assert reward_1 < 0
    assert info["number_phenotyped"] == 25
    assert env.observation_space.contains(observation)

    # PEV is invalid until the minimum training size has been reached.
    observation, invalid_reward, terminated, truncated, info = env.step(
        PhenotypingAction.HIGHEST_PEV
    )

    assert not terminated
    assert info["event"] == "invalid_action"
    assert invalid_reward < 0
    assert info["number_phenotyped"] == 25

    # The second valid batch reaches the 50-candidate budget and forces
    # generation finalization.
    observation, reward_2, terminated, truncated, info = env.step(
        PhenotypingAction.RANDOM
    )

    assert terminated
    assert not truncated
    assert env.observation_space.contains(observation)
    assert info["event"] == "generation_finalized"
    assert info["completed_generation"] == 1
    assert "cycle_summary" in info
    assert bridge.generation == 2
    assert bridge.population_size == 1000

    cycle_summary = info["cycle_summary"]

    assert int(cycle_summary.loc[0, "number_phenotyped"]) == 50
    assert np.isfinite(
        cycle_summary.loc[0, "realized_genetic_gain"]
    )

    print("\nFinal environment render:")
    print(env.render())
    print("\nCycle summary:")
    print(cycle_summary.to_string(index=False))
    print("\nAll breeding-environment checks passed.")


if __name__ == "__main__":
    main()
