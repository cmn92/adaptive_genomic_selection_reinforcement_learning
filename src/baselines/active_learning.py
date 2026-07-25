"""
active_learning.py

Prediction-error-variance active-learning baseline.

Within each breeding generation, this strategy:

1. Selects an initial genetically diverse batch.
2. Phenotypes that initial batch.
3. Fits G-BLUP and calculates prediction error variance (PEV) for every
   unphenotyped candidate.
4. Selects the remaining candidates with the largest PEV.
5. Phenotypes the second batch.
6. Fits the final AlphaSimR RR-BLUP model using all phenotyped candidates.
7. Selects parents and creates the next candidate generation.

The strategy uses formal model-based uncertainty rather than marker
distance alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from src.baselines.diversity_sampling import (
    DiversitySamplingStrategy,
)
from src.environment.r_bridge import (
    RBreedingBridge,
    RBridgeError,
)


@dataclass
class ActiveLearningGenerationResult:
    """Outputs from one active-learning breeding generation."""

    generation: int
    initial_indices: np.ndarray
    uncertainty_indices: np.ndarray
    all_phenotyped_indices: np.ndarray
    initial_model_accuracy: float
    final_model_accuracy: float
    uncertainty_table: pd.DataFrame
    cycle_summary: pd.DataFrame
    phenotype_table: pd.DataFrame
    prediction_table: pd.DataFrame
    selection_table: pd.DataFrame
    selection_seconds: float
    cycle_seconds: float
    next_population_size: int


class ActiveLearningStrategy:
    """
    Two-stage PEV-based active phenotyping strategy.

    Parameters
    ----------
    initial_batch_size:
        Number of candidates phenotyped before PEV is calculated.

    initial_strategy:
        Strategy used to construct the first training batch. The default
        is diversity sampling using greedy maximin selection.

    name:
        Name used in experiment outputs.
    """

    def __init__(
        self,
        initial_batch_size: int = 50,
        initial_strategy: DiversitySamplingStrategy | None = None,
        name: str = "active_learning_pev",
    ) -> None:
        if (
            isinstance(initial_batch_size, bool)
            or not isinstance(initial_batch_size, (int, np.integer))
            or initial_batch_size < 2
        ):
            raise ValueError(
                "'initial_batch_size' must be an integer of at least 2."
            )

        if not isinstance(name, str) or not name.strip():
            raise ValueError("'name' must be a non-empty string.")

        if initial_strategy is None:
            initial_strategy = DiversitySamplingStrategy(
                initial_selection="centroid_farthest",
                standardize_markers=True,
                sort_indices=True,
            )

        if not isinstance(
            initial_strategy,
            DiversitySamplingStrategy,
        ):
            raise TypeError(
                "'initial_strategy' must be a "
                "DiversitySamplingStrategy instance."
            )

        self.initial_batch_size = int(initial_batch_size)
        self.initial_strategy = initial_strategy
        self.name = name.strip()

    def run_generation(
        self,
        *,
        bridge: RBreedingBridge,
        number_to_phenotype: int,
        rng: np.random.Generator,
        number_of_parents: int = 20,
        number_of_crosses: int = 100,
        f1_per_cross: int = 1,
        dh_per_f1: int = 10,
        reps: int = 1,
        trait: int = 1,
        snp_chip: int = 1,
        n_cores: int = 1,
        seed: int | None = None,
    ) -> ActiveLearningGenerationResult:
        """
        Run one complete PEV-based active-learning generation.

        Parameters
        ----------
        bridge:
            Initialized R breeding bridge.

        number_to_phenotype:
            Total phenotyping budget for this generation.

        rng:
            NumPy random-number generator.

        Remaining arguments:
            Fixed breeding and genomic-prediction settings.

        Returns
        -------
        ActiveLearningGenerationResult
            Complete active-learning and breeding outputs.
        """
        if not isinstance(bridge, RBreedingBridge):
            raise TypeError(
                "'bridge' must be an RBreedingBridge instance."
            )

        if not isinstance(rng, np.random.Generator):
            raise TypeError(
                "'rng' must be a numpy.random.Generator."
            )

        number_to_phenotype = self._validate_positive_integer(
            number_to_phenotype,
            "number_to_phenotype",
        )

        if number_to_phenotype > bridge.population_size:
            raise ValueError(
                "'number_to_phenotype' cannot exceed the "
                f"population size of {bridge.population_size}."
            )

        if self.initial_batch_size >= number_to_phenotype:
            raise ValueError(
                "'initial_batch_size' must be smaller than the total "
                "'number_to_phenotype'."
            )

        number_of_parents = self._validate_positive_integer(
            number_of_parents,
            "number_of_parents",
        )
        number_of_crosses = self._validate_positive_integer(
            number_of_crosses,
            "number_of_crosses",
        )
        f1_per_cross = self._validate_positive_integer(
            f1_per_cross,
            "f1_per_cross",
        )
        dh_per_f1 = self._validate_positive_integer(
            dh_per_f1,
            "dh_per_f1",
        )
        reps = self._validate_positive_integer(reps, "reps")
        trait = self._validate_positive_integer(trait, "trait")
        snp_chip = self._validate_positive_integer(
            snp_chip,
            "snp_chip",
        )
        n_cores = self._validate_positive_integer(
            n_cores,
            "n_cores",
        )

        if seed is None:
            generation_seed = (
                bridge.base_seed + bridge.generation - 1
            )
        else:
            generation_seed = self._validate_positive_integer(
                seed,
                "seed",
            )

        generation = bridge.generation
        cycle_start = perf_counter()

        bridge.start_generation()

        # --------------------------------------------------------------
        # Stage 1: construct and phenotype the initial diverse batch
        # --------------------------------------------------------------

        candidate_data = bridge.get_candidate_data(
            include_markers=True
        )

        selection_start = perf_counter()

        initial_indices = self.initial_strategy.select(
            candidate_data=candidate_data,
            number_to_phenotype=self.initial_batch_size,
            rng=rng,
        )

        bridge.phenotype_batch(
            selected_indices=initial_indices,
            reps=reps,
            seed=generation_seed,
        )

        initial_model_result = bridge.fit_current_model(
            trait=trait
        )

        initial_model_accuracy = float(
            initial_model_result["prediction_accuracy"]
        )

        # --------------------------------------------------------------
        # Stage 2: calculate formal PEV and select the most uncertain
        # --------------------------------------------------------------

        second_batch_size = (
            number_to_phenotype - self.initial_batch_size
        )

        selected_by_uncertainty = True

        try:
            uncertainty_result = bridge.compute_current_uncertainty(
                trait=trait,
                snp_chip=snp_chip,
                n_cores=n_cores,
            )

            unphenotyped_uncertainty = uncertainty_result[
                "unphenotyped_uncertainty"
            ].copy()

            if len(unphenotyped_uncertainty) < second_batch_size:
                raise RuntimeError(
                    "There are not enough unphenotyped candidates to "
                    "complete the active-learning batch."
                )

            required_columns = {
                "population_index",
                "individual_id",
                "prediction_error_variance",
            }

            missing_columns = required_columns.difference(
                unphenotyped_uncertainty.columns
            )

            if missing_columns:
                raise RuntimeError(
                    "The uncertainty table is missing required columns: "
                    + ", ".join(sorted(missing_columns))
                )

            ranked_second_batch = (
                unphenotyped_uncertainty
                .sort_values(
                    "prediction_error_variance",
                    ascending=False,
                    kind="stable",
                )
                .head(second_batch_size)
                .copy()
            )

            # R returns one-based population indices. Python uses zero-based.
            uncertainty_indices = (
                ranked_second_batch["population_index"]
                .to_numpy(dtype=np.int64)
                - 1
            )

        except RBridgeError:
            selected_by_uncertainty = False
            unphenotyped_indices = bridge.get_unphenotyped_indices()

            if unphenotyped_indices.size < second_batch_size:
                raise RuntimeError(
                    "There are not enough unphenotyped candidates to "
                    "complete the active-learning batch."
                )

            candidate_ids = bridge.get_candidate_ids()
            uncertainty_indices = rng.choice(
                unphenotyped_indices,
                size=second_batch_size,
                replace=False,
            ).astype(np.int64)

            unphenotyped_uncertainty = pd.DataFrame(
                {
                    "population_index": (
                        unphenotyped_indices + 1
                    ),
                    "individual_id": [
                        candidate_ids[index]
                        for index in unphenotyped_indices
                    ],
                    "prediction_error_variance": np.nan,
                    "reliability": np.nan,
                    "uncertainty_rank": np.nan,
                    "selection_source": (
                        "random_fallback_no_pev"
                    ),
                }
            )

            ranked_second_batch = unphenotyped_uncertainty[
                unphenotyped_uncertainty[
                    "population_index"
                ].isin(uncertainty_indices + 1)
            ].copy()

        if "selection_source" not in unphenotyped_uncertainty.columns:
            unphenotyped_uncertainty[
                "selection_source"
            ] = "highest_pev"

        selected_mask = unphenotyped_uncertainty[
            "population_index"
        ].isin(uncertainty_indices + 1)
        unphenotyped_uncertainty.loc[
            selected_mask,
            "selected_for_second_batch",
        ] = True
        unphenotyped_uncertainty.loc[
            ~selected_mask,
            "selected_for_second_batch",
        ] = False

        if not selected_by_uncertainty:
            ranked_second_batch[
                "selected_for_second_batch"
            ] = True

        if np.unique(uncertainty_indices).size != second_batch_size:
            raise RuntimeError(
                "Second-batch selection produced duplicate "
                "candidate indices."
            )

        overlap = np.intersect1d(
            initial_indices,
            uncertainty_indices,
        )

        if overlap.size > 0:
            raise RuntimeError(
                "PEV selection included candidates already used in "
                "the initial batch."
            )

        selection_seconds = perf_counter() - selection_start

        # --------------------------------------------------------------
        # Stage 3: phenotype the high-PEV candidates and refit RR-BLUP
        # --------------------------------------------------------------

        bridge.phenotype_batch(
            selected_indices=uncertainty_indices,
            reps=reps,
            seed=generation_seed + 1,
        )

        final_model_result = bridge.fit_current_model(
            trait=trait
        )

        final_model_accuracy = float(
            final_model_result["prediction_accuracy"]
        )

        all_phenotyped_indices = (
            bridge.get_phenotyped_indices()
        )

        if all_phenotyped_indices.size != number_to_phenotype:
            raise RuntimeError(
                "The active-learning strategy did not use the complete "
                "phenotyping budget."
            )

        # --------------------------------------------------------------
        # Stage 4: parent selection, crossing and next generation
        # --------------------------------------------------------------

        final_result = bridge.finalize_generation(
            number_of_parents=number_of_parents,
            number_of_crosses=number_of_crosses,
            f1_per_cross=f1_per_cross,
            dh_per_f1=dh_per_f1,
            trait=trait,
            seed=generation_seed,
        )

        cycle_seconds = perf_counter() - cycle_start

        cycle_summary = final_result[
            "cycle_summary"
        ].copy()

        cycle_summary.insert(
            0,
            "strategy",
            self.name,
        )

        cycle_summary["initial_batch_size"] = (
            self.initial_batch_size
        )
        cycle_summary["uncertainty_batch_size"] = (
            second_batch_size
        )
        cycle_summary["selected_by_uncertainty"] = (
            selected_by_uncertainty
        )
        cycle_summary["initial_model_accuracy"] = (
            initial_model_accuracy
        )
        cycle_summary["final_model_accuracy"] = (
            final_model_accuracy
        )
        cycle_summary["selection_seconds"] = (
            selection_seconds
        )
        cycle_summary["cycle_seconds"] = cycle_seconds

        return ActiveLearningGenerationResult(
            generation=generation,
            initial_indices=initial_indices.copy(),
            uncertainty_indices=uncertainty_indices.copy(),
            all_phenotyped_indices=(
                all_phenotyped_indices.copy()
            ),
            initial_model_accuracy=initial_model_accuracy,
            final_model_accuracy=final_model_accuracy,
            uncertainty_table=unphenotyped_uncertainty,
            cycle_summary=cycle_summary,
            phenotype_table=final_result[
                "phenotype_table"
            ],
            prediction_table=final_result[
                "prediction_table"
            ],
            selection_table=final_result[
                "selection_table"
            ],
            selection_seconds=selection_seconds,
            cycle_seconds=cycle_seconds,
            next_population_size=final_result[
                "next_population_size"
            ],
        )

    @staticmethod
    def _validate_positive_integer(
        value: Any,
        name: str,
    ) -> int:
        """Validate a positive integer."""
        if isinstance(value, (bool, np.bool_)):
            raise TypeError(
                f"'{name}' must be a positive integer."
            )

        if not np.isscalar(value):
            raise TypeError(
                f"'{name}' must be a scalar positive integer."
            )

        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"'{name}' must be a positive integer."
            ) from exc

        if not np.isfinite(numeric_value):
            raise ValueError(f"'{name}' must be finite.")

        if numeric_value % 1 != 0 or numeric_value < 1:
            raise ValueError(
                f"'{name}' must be a positive integer."
            )

        return int(numeric_value)

    def __repr__(self) -> str:
        """Return a readable strategy representation."""
        return (
            f"{self.__class__.__name__}("
            f"name={self.name!r}, "
            f"initial_batch_size={self.initial_batch_size}, "
            f"initial_strategy={self.initial_strategy!r})"
        )
