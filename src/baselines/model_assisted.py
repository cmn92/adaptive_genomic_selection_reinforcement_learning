"""
model_assisted.py

Staged model-assisted phenotyping baselines.

These policies use the same first-stage diverse training batch as the PEV
active-learning baseline, then spend the remaining fixed budget according to
one model-derived score.
"""

from __future__ import annotations

from time import perf_counter

import numpy as np
import pandas as pd

from src.baselines.active_learning import (
    ActiveLearningGenerationResult,
    ActiveLearningStrategy,
)
from src.environment.r_bridge import RBreedingBridge


class HighestGEBVStrategy(ActiveLearningStrategy):
    """Phenotype an initial diverse batch, then highest predicted GEBV."""

    def __init__(
        self,
        initial_batch_size: int = 50,
        name: str = "highest_gebv",
    ) -> None:
        super().__init__(
            initial_batch_size=initial_batch_size,
            name=name,
        )

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
        """Run one generation with highest-GEBV second-stage selection."""
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

        if self.initial_batch_size >= number_to_phenotype:
            raise ValueError(
                "'initial_batch_size' must be smaller than the total "
                "'number_to_phenotype'."
            )

        generation_seed = (
            bridge.base_seed + bridge.generation - 1
            if seed is None
            else self._validate_positive_integer(
                seed,
                "seed",
            )
        )

        generation = bridge.generation
        cycle_start = perf_counter()

        bridge.start_generation()

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

        second_batch_size = (
            number_to_phenotype - self.initial_batch_size
        )

        prediction_table = initial_model_result[
            "prediction_table"
        ].copy()

        required_columns = {
            "population_index",
            "individual_id",
            "predicted_gebv",
        }
        missing_columns = required_columns.difference(
            prediction_table.columns
        )

        if missing_columns:
            raise RuntimeError(
                "The prediction table is missing required columns: "
                + ", ".join(sorted(missing_columns))
            )

        unphenotyped = set(
            (bridge.get_unphenotyped_indices() + 1).tolist()
        )
        unphenotyped_predictions = prediction_table[
            prediction_table["population_index"].isin(
                unphenotyped
            )
        ].copy()

        if len(unphenotyped_predictions) < second_batch_size:
            raise RuntimeError(
                "There are not enough unphenotyped candidates to "
                "complete the highest-GEBV batch."
            )

        ranked_second_batch = (
            unphenotyped_predictions.sort_values(
                "predicted_gebv",
                ascending=False,
                kind="stable",
            )
            .head(second_batch_size)
            .copy()
        )

        second_stage_indices = (
            ranked_second_batch["population_index"]
            .to_numpy(dtype=np.int64)
            - 1
        )

        if np.unique(second_stage_indices).size != second_batch_size:
            raise RuntimeError(
                "Highest-GEBV selection produced duplicate candidates."
            )

        if np.intersect1d(
            initial_indices,
            second_stage_indices,
        ).size:
            raise RuntimeError(
                "Highest-GEBV selection included candidates from the "
                "initial batch."
            )

        selection_seconds = perf_counter() - selection_start

        bridge.phenotype_batch(
            selected_indices=second_stage_indices,
            reps=reps,
            seed=generation_seed + 1,
        )

        final_model_result = bridge.fit_current_model(
            trait=trait
        )
        final_model_accuracy = float(
            final_model_result["prediction_accuracy"]
        )

        all_phenotyped_indices = bridge.get_phenotyped_indices()

        if all_phenotyped_indices.size != number_to_phenotype:
            raise RuntimeError(
                "The highest-GEBV strategy did not use the complete "
                "phenotyping budget."
            )

        final_result = bridge.finalize_generation(
            number_of_parents=number_of_parents,
            number_of_crosses=number_of_crosses,
            f1_per_cross=f1_per_cross,
            dh_per_f1=dh_per_f1,
            trait=trait,
            seed=generation_seed,
        )

        cycle_seconds = perf_counter() - cycle_start
        cycle_summary = final_result["cycle_summary"].copy()
        cycle_summary.insert(0, "strategy", self.name)
        cycle_summary["initial_batch_size"] = (
            self.initial_batch_size
        )
        cycle_summary["model_score_batch_size"] = (
            second_batch_size
        )
        cycle_summary["model_score"] = "predicted_gebv"
        cycle_summary["selected_by_uncertainty"] = False
        cycle_summary["initial_model_accuracy"] = (
            initial_model_accuracy
        )
        cycle_summary["final_model_accuracy"] = (
            final_model_accuracy
        )
        cycle_summary["selection_seconds"] = selection_seconds
        cycle_summary["cycle_seconds"] = cycle_seconds

        second_stage_table = unphenotyped_predictions.copy()
        second_stage_table["selected_for_second_batch"] = (
            second_stage_table["population_index"].isin(
                second_stage_indices + 1
            )
        )

        return ActiveLearningGenerationResult(
            generation=generation,
            initial_indices=initial_indices.copy(),
            uncertainty_indices=second_stage_indices.copy(),
            all_phenotyped_indices=(
                all_phenotyped_indices.copy()
            ),
            initial_model_accuracy=initial_model_accuracy,
            final_model_accuracy=final_model_accuracy,
            uncertainty_table=second_stage_table,
            cycle_summary=cycle_summary,
            phenotype_table=final_result["phenotype_table"],
            prediction_table=final_result["prediction_table"],
            selection_table=final_result["selection_table"],
            selection_seconds=selection_seconds,
            cycle_seconds=cycle_seconds,
            next_population_size=final_result[
                "next_population_size"
            ],
        )
