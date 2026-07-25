"""
Integration test for random sampling through the Python-R bridge.
"""

from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.random_sampling import RandomSamplingStrategy
from src.environment.r_bridge import RBreedingBridge


def main() -> None:
    """Run one breeding cycle using the random baseline."""

    bridge = RBreedingBridge(
        project_root=PROJECT_ROOT,
        population_file="data/initial_candidate_population.RData",
        seed=12345,
    )

    strategy = RandomSamplingStrategy(
        sort_indices=True
    )

    rng = np.random.default_rng(12345)

    candidate_data = bridge.get_candidate_data(
        include_markers=False
    )

    selected_indices = strategy.select(
        candidate_data=candidate_data,
        number_to_phenotype=200,
        rng=rng,
    )

    result = bridge.step(
        selected_indices=selected_indices,
        number_of_parents=20,
        number_of_crosses=100,
        f1_per_cross=1,
        dh_per_f1=10,
        reps=1,
        trait=1,
        seed=12345,
    )

    cycle_summary = result["cycle_summary"]

    print("\nRandom baseline completed one breeding cycle.")
    print("\nSelected indices, first ten:")
    print(selected_indices[:10])

    print("\nCycle summary:")
    print(cycle_summary.to_string(index=False))

    assert selected_indices.size == 200
    assert len(result["phenotype_table"]) == 200
    assert len(result["selection_table"]) == 20
    assert result["next_population_size"] == 1000

    assert int(
        cycle_summary.loc[0, "number_phenotyped"]
    ) == 200

    assert np.isfinite(
        cycle_summary.loc[0, "prediction_accuracy"]
    )

    assert np.isfinite(
        cycle_summary.loc[0, "realized_genetic_gain"]
    )

    print("\nAll random-strategy bridge checks passed.")


if __name__ == "__main__":
    main()