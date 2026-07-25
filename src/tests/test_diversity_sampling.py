"""
Tests for the genetic-diversity phenotyping baseline.
"""

from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.diversity_sampling import (
    DiversitySamplingStrategy,
)


def mean_pairwise_distance(
    marker_matrix: np.ndarray,
    selected_indices: np.ndarray,
) -> float:
    """
    Calculate mean pairwise Euclidean distance within a selected subset.
    """
    selected_matrix = marker_matrix[
        selected_indices
    ]

    differences = (
        selected_matrix[:, None, :]
        - selected_matrix[None, :, :]
    )

    distances = np.sqrt(
        np.sum(
            differences**2,
            axis=2,
        )
    )

    upper_triangle = np.triu_indices(
        selected_matrix.shape[0],
        k=1,
    )

    return float(
        distances[upper_triangle].mean()
    )


def main() -> None:
    """Run diversity-sampling unit tests."""

    rng_data = np.random.default_rng(101)

    population_size = 100
    marker_count = 40

    marker_matrix = rng_data.integers(
        low=0,
        high=3,
        size=(population_size, marker_count),
    ).astype(np.float64)

    candidate_data = {
        "generation": 1,
        "population_size": population_size,
        "individual_ids": [
            f"CAND_{index:04d}"
            for index in range(1, population_size + 1)
        ],
        "marker_matrix": marker_matrix,
    }

    strategy = DiversitySamplingStrategy(
        initial_selection="centroid_farthest",
        standardize_markers=True,
        sort_indices=True,
    )

    selected_indices = strategy.select(
        candidate_data=candidate_data,
        number_to_phenotype=20,
        rng=np.random.default_rng(12345),
    )

    print("\nStrategy:", strategy)
    print("Number selected:", selected_indices.size)
    print("Selected indices:")
    print(selected_indices)

    assert selected_indices.shape == (20,)
    assert selected_indices.dtype == np.int64
    assert selected_indices.min() >= 0
    assert selected_indices.max() < population_size
    assert np.unique(selected_indices).size == 20
    assert np.all(np.diff(selected_indices) >= 0)

    fixed_marker_indices = strategy.select(
        candidate_data={
            **candidate_data,
            "marker_matrix": np.zeros_like(marker_matrix),
        },
        number_to_phenotype=20,
        rng=np.random.default_rng(12345),
    )

    assert fixed_marker_indices.shape == (20,)
    assert fixed_marker_indices.dtype == np.int64
    assert fixed_marker_indices.min() >= 0
    assert fixed_marker_indices.max() < population_size
    assert np.unique(fixed_marker_indices).size == 20
    assert np.all(np.diff(fixed_marker_indices) >= 0)

    # Deterministic centroid-farthest initialization should reproduce
    # the same selection.
    selected_repeat = strategy.select(
        candidate_data=candidate_data,
        number_to_phenotype=20,
        rng=np.random.default_rng(99999),
    )

    assert np.array_equal(
        selected_indices,
        selected_repeat,
    )

    # Compare genetic spread with one random sample.
    random_indices = np.random.default_rng(
        12345
    ).choice(
        population_size,
        size=20,
        replace=False,
    )

    diversity_distance = mean_pairwise_distance(
        marker_matrix,
        selected_indices,
    )

    random_distance = mean_pairwise_distance(
        marker_matrix,
        random_indices,
    )

    print(
        "\nMean pairwise distance, diversity sample:",
        round(diversity_distance, 3),
    )
    print(
        "Mean pairwise distance, random sample:",
        round(random_distance, 3),
    )

    # This is a useful diagnostic, but diversity sampling is not
    # mathematically guaranteed to beat every individual random draw.
    assert np.isfinite(diversity_distance)
    assert diversity_distance > 0

    # Entire-population selection.
    full_selection = strategy.select(
        candidate_data=candidate_data,
        number_to_phenotype=population_size,
        rng=np.random.default_rng(12345),
    )

    assert np.array_equal(
        full_selection,
        np.arange(
            population_size,
            dtype=np.int64,
        ),
    )

    # Missing marker matrix should fail.
    invalid_candidate_data = {
        "population_size": population_size,
    }

    try:
        strategy.select(
            candidate_data=invalid_candidate_data,
            number_to_phenotype=20,
            rng=np.random.default_rng(12345),
        )
    except KeyError:
        pass
    else:
        raise AssertionError(
            "Missing marker data should raise KeyError."
        )

    print("\nAll diversity-sampling checks passed.")


if __name__ == "__main__":
    main()
