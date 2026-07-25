"""
breeding_env.py

Gymnasium environment for reinforcement learning in adaptive phenotyping.

One episode contains several breeding generations. Within each generation,
the agent repeatedly chooses a batch-selection action or stops and finalizes
the generation.

Actions
-------
0 RANDOM
1 DIVERSITY
2 HIGHEST_PEV
3 HIGHEST_GEBV
4 STOP

Observations contain breeder-visible model, uncertainty, budget, and historical
summary information. Current true breeding values are hidden.

Rewards use hidden simulator outcomes after generation finalization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:
    raise ImportError(
        "gymnasium is required for BreedingEnv. Install it with:\n"
        "python -m pip install gymnasium"
    ) from exc

from src.environment.actions import (
    PhenotypingAction,
    action_name,
    build_action_mask,
    validate_action,
)
from src.environment.r_bridge import RBreedingBridge, RBridgeError
from src.environment.reward import (
    RewardBreakdown,
    RewardConfig,
    batch_cost_reward,
    final_generation_reward,
    invalid_action_reward,
    model_quality_reward,
)
from src.environment.state import (
    BreedingStateSnapshot,
    build_observation,
    observation_size,
)


@dataclass(frozen=True)
class BreedingEnvConfig:
    """Configuration for the adaptive phenotyping environment."""

    maximum_generations: int = 10
    batch_size: int = 25
    minimum_training_size: int = 50
    maximum_phenotypes: int = 200
    number_of_parents: int = 20
    number_of_crosses: int = 100
    f1_per_cross: int = 1
    dh_per_f1: int = 10
    reps: int = 1
    trait: int = 1
    snp_chip: int = 1
    n_cores: int = 1
    seed: int = 12345

    def __post_init__(self) -> None:
        fields = self.__dict__

        for name, value in fields.items():
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, np.integer)
            ):
                raise TypeError(f"'{name}' must be an integer.")
            if value < 1:
                raise ValueError(f"'{name}' must be at least 1.")

        if self.minimum_training_size > self.maximum_phenotypes:
            raise ValueError(
                "'minimum_training_size' cannot exceed "
                "'maximum_phenotypes'."
            )

        if self.maximum_phenotypes % self.batch_size != 0:
            raise ValueError(
                "'maximum_phenotypes' must be divisible by 'batch_size'."
            )

        if self.minimum_training_size % self.batch_size != 0:
            raise ValueError(
                "'minimum_training_size' must be divisible by 'batch_size'."
            )


class BreedingEnv(gym.Env):
    """
    Gymnasium environment for sequential adaptive phenotyping.

    Notes
    -----
    The environment automatically refits RR-BLUP after each batch once the
    minimum training size is reached. It also refreshes PEV estimates so the
    HIGHEST_PEV action is model-based.

    Invalid actions receive a penalty and do not change the breeding state.
    The current valid-action mask is returned in the info dictionary.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        *,
        bridge: RBreedingBridge,
        config: BreedingEnvConfig | None = None,
        reward_config: RewardConfig | None = None,
    ) -> None:
        super().__init__()

        if not isinstance(bridge, RBreedingBridge):
            raise TypeError(
                "'bridge' must be an RBreedingBridge instance."
            )

        self.bridge = bridge
        self.config = config or BreedingEnvConfig()
        self.reward_config = reward_config or RewardConfig()

        if self.config.maximum_phenotypes > bridge.population_size:
            raise ValueError(
                "'maximum_phenotypes' cannot exceed population size."
            )

        self.action_space = spaces.Discrete(
            len(PhenotypingAction)
        )
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(observation_size(),),
            dtype=np.float32,
        )

        self._rng = np.random.default_rng(
            self.config.seed
        )

        self._marker_matrix: np.ndarray | None = None
        self._prediction_table = None
        self._uncertainty_table = None
        self._last_uncertainty_error: str | None = None

        self._previous_genetic_gain = 0.0
        self._previous_variance_retention = 1.0
        self._previous_mean_reliability = 0.0
        self._initial_generation_variance: float | None = None

        self._episode_return = 0.0
        self._episode_steps = 0
        self._terminated = False
        self._last_info: dict[str, Any] = {}

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the breeding program and begin generation one."""
        super().reset(seed=seed)

        reset_seed = (
            self.config.seed
            if seed is None
            else int(seed)
        )

        self._rng = np.random.default_rng(
            reset_seed
        )

        self.bridge.reset(seed=reset_seed)
        self.bridge.start_generation()

        self._marker_matrix = self.bridge.get_marker_matrix(
            snp_chip=self.config.snp_chip
        )
        self._prediction_table = None
        self._uncertainty_table = None
        self._last_uncertainty_error = None

        self._previous_genetic_gain = 0.0
        self._previous_variance_retention = 1.0
        self._previous_mean_reliability = 0.0
        self._initial_generation_variance = None

        self._episode_return = 0.0
        self._episode_steps = 0
        self._terminated = False

        observation = self._build_current_observation()
        info = self._build_info(
            event="reset",
            reward_breakdown=None,
        )
        self._last_info = info

        return observation, info

    def step(
        self,
        action: int,
    ) -> tuple[
        np.ndarray,
        float,
        bool,
        bool,
        dict[str, Any],
    ]:
        """Apply one phenotyping or finalization action."""
        if self._terminated:
            raise RuntimeError(
                "The episode has terminated. Call reset() before step()."
            )

        validated_action = validate_action(action)
        mask = self.action_masks()

        if not bool(mask[int(validated_action)]):
            breakdown = invalid_action_reward(
                self.reward_config
            )
            reward = breakdown.total
            self._episode_return += reward
            self._episode_steps += 1

            observation = self._build_current_observation()
            info = self._build_info(
                event="invalid_action",
                reward_breakdown=breakdown,
                selected_indices=None,
                action=validated_action,
            )
            self._last_info = info

            return observation, reward, False, False, info

        if validated_action == PhenotypingAction.STOP:
            return self._finalize_generation(
                action=validated_action
            )

        selected_indices = self._select_batch(
            validated_action
        )

        self.bridge.phenotype_batch(
            selected_indices=selected_indices,
            reps=self.config.reps,
            seed=(
                self.config.seed
                + self._episode_steps
                + self.bridge.generation
            ),
        )

        if (
            self.bridge.number_currently_phenotyped
            >= self.config.minimum_training_size
        ):
            self._refresh_model_and_uncertainty()
            model_quality_breakdown = (
                self._model_quality_breakdown()
            )
        else:
            self._prediction_table = None
            self._uncertainty_table = None
            self._last_uncertainty_error = None
            model_quality_breakdown = (
                model_quality_reward(
                    previous_mean_reliability=0.0,
                    current_mean_reliability=0.0,
                    config=self.reward_config,
                )
            )

        batch_breakdown = batch_cost_reward(
            batch_size=self.config.batch_size,
            maximum_phenotypes=(
                self.config.maximum_phenotypes
            ),
            config=self.reward_config,
        )

        breakdown = self._combine_reward_breakdowns(
            batch_breakdown,
            model_quality_breakdown,
        )

        reward = breakdown.total
        self._episode_return += reward
        self._episode_steps += 1

        # Force finalization once the maximum budget is exhausted.
        if (
            self.bridge.number_currently_phenotyped
            >= self.config.maximum_phenotypes
        ):
            return self._finalize_generation(
                action=validated_action,
                selected_indices=selected_indices,
                already_charged_batch_reward=reward,
            )

        observation = self._build_current_observation()
        info = self._build_info(
            event="phenotype_batch",
            reward_breakdown=breakdown,
            selected_indices=selected_indices,
            action=validated_action,
        )
        self._last_info = info

        return observation, reward, False, False, info

    def action_masks(self) -> np.ndarray:
        """Return the current valid-action mask."""
        return build_action_mask(
            population_size=self.bridge.population_size,
            number_phenotyped=(
                self.bridge.number_currently_phenotyped
            ),
            batch_size=self.config.batch_size,
            maximum_phenotypes=(
                self.config.maximum_phenotypes
            ),
            minimum_training_size=(
                self.config.minimum_training_size
            ),
            model_available=(
                self._prediction_table is not None
            ),
            uncertainty_available=(
                self._uncertainty_table is not None
            ),
        )

    def _refresh_model_and_uncertainty(self) -> None:
        """
        Fit the current prediction model and refresh PEV estimates if possible.

        In late generations the relationship matrix used by the R-side
        uncertainty calculation can become singular or nonfinite. Keep the
        fitted GEBV model available, but mark formal PEV unavailable so the
        action mask disables HIGHEST_PEV instead of aborting training.
        """
        model_result = self.bridge.fit_current_model(
            trait=self.config.trait
        )
        self._prediction_table = model_result[
            "prediction_table"
        ]

        if not self._has_usable_marker_variation():
            self._uncertainty_table = None
            self._last_uncertainty_error = (
                "No usable polymorphic markers are available for "
                "formal PEV calculation."
            )
            return

        try:
            uncertainty_result = (
                self.bridge.compute_current_uncertainty(
                    trait=self.config.trait,
                    snp_chip=self.config.snp_chip,
                    n_cores=self.config.n_cores,
                )
            )
        except RBridgeError as exc:
            self._uncertainty_table = None
            self._last_uncertainty_error = str(exc)
            return

        self._uncertainty_table = uncertainty_result[
            "uncertainty_table"
        ]
        self._last_uncertainty_error = None

    def _model_quality_breakdown(self) -> RewardBreakdown:
        """Return reliability-improvement shaping for the latest model."""
        if self._uncertainty_table is None:
            return model_quality_reward(
                previous_mean_reliability=(
                    self._previous_mean_reliability
                ),
                current_mean_reliability=(
                    self._previous_mean_reliability
                ),
                config=self.reward_config,
            )

        _, _, current_mean_reliability = (
            self._uncertainty_summaries()
        )

        breakdown = model_quality_reward(
            previous_mean_reliability=(
                self._previous_mean_reliability
            ),
            current_mean_reliability=(
                current_mean_reliability
            ),
            config=self.reward_config,
        )

        self._previous_mean_reliability = (
            current_mean_reliability
        )

        return breakdown

    @staticmethod
    def _combine_reward_breakdowns(
        first: RewardBreakdown,
        second: RewardBreakdown,
    ) -> RewardBreakdown:
        """Add two reward breakdowns component-wise."""
        return RewardBreakdown(
            total=first.total + second.total,
            genetic_gain_component=(
                first.genetic_gain_component
                + second.genetic_gain_component
            ),
            variance_component=(
                first.variance_component
                + second.variance_component
            ),
            cost_component=(
                first.cost_component
                + second.cost_component
            ),
            model_quality_component=(
                first.model_quality_component
                + second.model_quality_component
            ),
            invalid_action_component=(
                first.invalid_action_component
                + second.invalid_action_component
            ),
        )

    def _has_usable_marker_variation(self) -> bool:
        """Return whether the current marker matrix can support PEV fitting."""
        if self._marker_matrix is None:
            return False

        markers = np.asarray(
            self._marker_matrix,
            dtype=np.float64,
        )

        if markers.ndim != 2 or markers.shape[1] == 0:
            return False

        marker_variance = markers.var(axis=0)
        return bool(
            np.any(
                np.isfinite(marker_variance)
                & (marker_variance > 0.0)
            )
        )

    def _select_batch(
        self,
        action: PhenotypingAction,
    ) -> np.ndarray:
        """Select one zero-based batch according to the chosen action."""
        unphenotyped = self.bridge.get_unphenotyped_indices()

        if action == PhenotypingAction.RANDOM:
            return np.sort(
                self._rng.choice(
                    unphenotyped,
                    size=self.config.batch_size,
                    replace=False,
                ).astype(np.int64)
            )

        if action == PhenotypingAction.DIVERSITY:
            return np.sort(
                self._select_diverse_batch(
                    unphenotyped
                )
            )

        if action == PhenotypingAction.HIGHEST_PEV:
            if self._uncertainty_table is None:
                raise RuntimeError(
                    "PEV action requested without uncertainty estimates."
                )

            table = self._uncertainty_table
            available = table[
                ~table["is_phenotyped"].astype(bool)
            ].sort_values(
                "prediction_error_variance",
                ascending=False,
                kind="stable",
            )

            selected = (
                available["population_index"]
                .head(self.config.batch_size)
                .to_numpy(dtype=np.int64)
                - 1
            )
            return np.sort(selected)

        if action == PhenotypingAction.HIGHEST_GEBV:
            if self._prediction_table is None:
                raise RuntimeError(
                    "GEBV action requested without fitted predictions."
                )

            table = self._prediction_table.copy()
            table = table[
                table["population_index"].isin(
                    unphenotyped + 1
                )
            ]

            selected = (
                table.sort_values(
                    "predicted_gebv",
                    ascending=False,
                    kind="stable",
                )["population_index"]
                .head(self.config.batch_size)
                .to_numpy(dtype=np.int64)
                - 1
            )
            return np.sort(selected)

        raise RuntimeError(
            f"Action {action!r} does not select a batch."
        )

    def _select_diverse_batch(
        self,
        unphenotyped: np.ndarray,
    ) -> np.ndarray:
        """
        Greedy maximin selection relative to the current training set.

        At the start of a generation, the first candidate is chosen farthest
        from the population centroid. Later batches prioritize candidates far
        from all candidates already phenotyped in the generation.
        """
        if self._marker_matrix is None:
            raise RuntimeError("Marker matrix is unavailable.")

        markers = np.asarray(
            self._marker_matrix,
            dtype=np.float64,
        )

        marker_variance = markers.var(axis=0)
        polymorphic = (
            np.isfinite(marker_variance)
            & (marker_variance > 0.0)
        )

        if not np.any(polymorphic):
            return np.sort(
                self._rng.choice(
                    unphenotyped,
                    size=self.config.batch_size,
                    replace=False,
                ).astype(np.int64)
            )

        markers = markers[:, polymorphic]
        means = markers.mean(axis=0)
        standard_deviations = markers.std(axis=0)
        usable = (
            np.isfinite(standard_deviations)
            & (standard_deviations > 0.0)
        )

        if not np.any(usable):
            return np.sort(
                self._rng.choice(
                    unphenotyped,
                    size=self.config.batch_size,
                    replace=False,
                ).astype(np.int64)
            )

        markers = markers[:, usable]
        means = means[usable]
        standard_deviations = (
            standard_deviations[usable]
        )
        markers = (
            markers - means
        ) / standard_deviations

        selected_existing = (
            self.bridge.get_phenotyped_indices()
        )

        chosen: list[int] = []
        available = unphenotyped.copy()

        if selected_existing.size == 0:
            centroid = markers.mean(axis=0)
            distance = np.sum(
                (markers[available] - centroid) ** 2,
                axis=1,
            )
            first = int(available[np.argmax(distance)])
            chosen.append(first)
        else:
            reference = markers[selected_existing]
            candidate_matrix = markers[available]
            squared = np.sum(
                (
                    candidate_matrix[:, None, :]
                    - reference[None, :, :]
                )
                ** 2,
                axis=2,
            )
            minimum_distance = squared.min(axis=1)
            first = int(
                available[np.argmax(minimum_distance)]
            )
            chosen.append(first)

        while len(chosen) < self.config.batch_size:
            excluded = np.isin(
                available,
                np.asarray(chosen, dtype=np.int64),
            )
            candidates = available[~excluded]

            reference_indices = np.concatenate(
                [
                    selected_existing,
                    np.asarray(chosen, dtype=np.int64),
                ]
            )

            squared = np.sum(
                (
                    markers[candidates][:, None, :]
                    - markers[reference_indices][None, :, :]
                )
                ** 2,
                axis=2,
            )
            minimum_distance = squared.min(axis=1)
            next_index = int(
                candidates[np.argmax(minimum_distance)]
            )
            chosen.append(next_index)

        return np.asarray(chosen, dtype=np.int64)

    def _training_marker_diversity(self) -> float:
        """
        Calculate mean Euclidean distance within the current training set.

        To keep this inexpensive, at most 100 phenotyped candidates are used.
        """
        indices = self.bridge.get_phenotyped_indices()

        if indices.size < 2 or self._marker_matrix is None:
            return 0.0

        if indices.size > 100:
            indices = indices[:100]

        matrix = np.asarray(
            self._marker_matrix[indices],
            dtype=np.float64,
        )

        differences = (
            matrix[:, None, :]
            - matrix[None, :, :]
        )
        distances = np.sqrt(
            np.sum(differences**2, axis=2)
        )

        upper = np.triu_indices(
            indices.size,
            k=1,
        )

        return float(
            distances[upper].mean()
        )

    def _candidate_marker_diversity(self) -> float:
        """
        Calculate mean marker distance in the current candidate population.

        The value is a breeder-visible diversity summary of the population
        currently being sampled. To keep observations inexpensive, the first
        100 candidates are used when the population is larger than that.
        """
        if self._marker_matrix is None:
            return 0.0

        sample_size = min(
            100,
            int(self._marker_matrix.shape[0]),
        )

        if sample_size < 2:
            return 0.0

        matrix = np.asarray(
            self._marker_matrix[:sample_size],
            dtype=np.float64,
        )

        differences = (
            matrix[:, None, :]
            - matrix[None, :, :]
        )
        distances = np.sqrt(
            np.sum(differences**2, axis=2)
        )

        upper = np.triu_indices(
            sample_size,
            k=1,
        )

        return float(
            distances[upper].mean()
        )

    def _uncertainty_summaries(
        self,
    ) -> tuple[float, float, float]:
        """Return mean PEV, maximum PEV, and mean reliability."""
        if self._uncertainty_table is None:
            return 0.0, 0.0, 0.0

        table = self._uncertainty_table
        available = table[
            ~table["is_phenotyped"].astype(bool)
        ]

        if available.empty:
            return 0.0, 0.0, 0.0

        return (
            float(
                available[
                    "prediction_error_variance"
                ].mean()
            ),
            float(
                available[
                    "prediction_error_variance"
                ].max()
            ),
            float(
                available["reliability"].mean()
            ),
        )

    def _prediction_summaries(
        self,
    ) -> tuple[float, float]:
        """Return mean and standard deviation of current predicted GEBVs."""
        if self._prediction_table is None:
            return 0.0, 0.0

        values = self._prediction_table[
            "predicted_gebv"
        ].to_numpy(dtype=float)

        return (
            float(np.mean(values)),
            float(np.std(values, ddof=1)),
        )

    def _prediction_residual_summaries(
        self,
    ) -> tuple[float, float, float]:
        """
        Return MAE, RMSE, and bias for observed phenotypes.

        These diagnostics compare current predictions against phenotypes
        collected in the active generation. They avoid hidden true breeding
        values while still exposing model-error information to the agent.
        """
        if self._prediction_table is None:
            return 0.0, 0.0, 0.0

        phenotype_table = (
            self.bridge.get_combined_phenotype_table()
        )

        if phenotype_table.empty:
            return 0.0, 0.0, 0.0

        phenotype_column = (
            f"phenotype_trait_{self.config.trait}"
        )

        required_columns = {
            "population_index",
            phenotype_column,
        }

        if not required_columns.issubset(
            phenotype_table.columns
        ):
            return 0.0, 0.0, 0.0

        if not {
            "population_index",
            "predicted_gebv",
        }.issubset(self._prediction_table.columns):
            return 0.0, 0.0, 0.0

        observed = phenotype_table[
            ["population_index", phenotype_column]
        ].copy()
        predicted = self._prediction_table[
            ["population_index", "predicted_gebv"]
        ]

        merged = observed.merge(
            predicted,
            on="population_index",
            how="inner",
        )

        if merged.empty:
            return 0.0, 0.0, 0.0

        residuals = (
            merged["predicted_gebv"].to_numpy(dtype=float)
            - merged[phenotype_column].to_numpy(dtype=float)
        )

        residuals = residuals[
            np.isfinite(residuals)
        ]

        if residuals.size == 0:
            return 0.0, 0.0, 0.0

        return (
            float(np.mean(np.abs(residuals))),
            float(
                np.sqrt(
                    np.mean(residuals**2)
                )
            ),
            float(np.mean(residuals)),
        )

    def _build_current_observation(
        self,
    ) -> np.ndarray:
        """Construct the current normalized observation."""

        mean_pev, max_pev, mean_reliability = (
            self._uncertainty_summaries()
        )

        mean_gebv, gebv_spread = (
            self._prediction_summaries()
        )

        residual_mae, residual_rmse, residual_bias = (
            self._prediction_residual_summaries()
        )

        # After the final breeding generation is completed, the bridge has
        # already advanced to the next population. Keep the terminal
        # observation within the configured episode horizon.
        observation_generation = min(
            self.bridge.generation,
            self.config.maximum_generations,
        )

        snapshot = BreedingStateSnapshot(
            generation=observation_generation,
            maximum_generations=(
                self.config.maximum_generations
            ),
            number_phenotyped=(
                self.bridge.number_currently_phenotyped
            ),
            maximum_phenotypes=(
                self.config.maximum_phenotypes
            ),
            model_available=(
                self._prediction_table is not None
            ),
            mean_pev=mean_pev,
            max_pev=max_pev,
            mean_reliability=mean_reliability,
            mean_predicted_gebv=mean_gebv,
            predicted_gebv_standard_deviation=(
                gebv_spread
            ),
            candidate_marker_diversity=(
                self._candidate_marker_diversity()
            ),
            training_marker_diversity=(
                self._training_marker_diversity()
            ),
            prediction_residual_mae=(
                residual_mae
            ),
            prediction_residual_rmse=(
                residual_rmse
            ),
            prediction_residual_bias=(
                residual_bias
            ),
            previous_genetic_gain=(
                self._previous_genetic_gain
            ),
            previous_variance_retention=(
                self._previous_variance_retention
            ),
        )

        return build_observation(snapshot)

    def _finalize_generation(
        self,
        *,
        action: PhenotypingAction,
        selected_indices: np.ndarray | None = None,
        already_charged_batch_reward: float = 0.0,
    ) -> tuple[
        np.ndarray,
        float,
        bool,
        bool,
        dict[str, Any],
    ]:
        """Finalize the current generation and start the next one."""
        if self._prediction_table is None:
            model_result = self.bridge.fit_current_model(
                trait=self.config.trait
            )
            self._prediction_table = model_result[
                "prediction_table"
            ]

        final_result = self.bridge.finalize_generation(
            number_of_parents=self.config.number_of_parents,
            number_of_crosses=self.config.number_of_crosses,
            f1_per_cross=self.config.f1_per_cross,
            dh_per_f1=self.config.dh_per_f1,
            trait=self.config.trait,
            seed=(
                self.config.seed
                + self.bridge.generation
                - 1
            ),
        )

        summary = final_result[
            "cycle_summary"
        ].iloc[0]

        current_initial_variance = float(
            summary["population_variance_gv_before"]
        )
        final_variance = float(
            summary["next_generation_variance_gv"]
        )

        if self._initial_generation_variance is None:
            self._initial_generation_variance = (
                current_initial_variance
            )

        variance_retention = (
            final_variance
            / self._initial_generation_variance
            if self._initial_generation_variance > 0
            else 0.0
        )

        complete_breakdown = final_generation_reward(
            realized_genetic_gain=float(
                summary["realized_genetic_gain"]
            ),
            variance_retention=variance_retention,
            number_phenotyped=int(
                summary["number_phenotyped"]
            ),
            maximum_phenotypes=(
                self.config.maximum_phenotypes
            ),
            config=self.reward_config,
        )

        # Batch costs have already been charged on previous phenotype steps.
        terminal_reward = (
            complete_breakdown.total
            - complete_breakdown.cost_component
        )

        reward = (
            already_charged_batch_reward
            + terminal_reward
        )

        self._previous_genetic_gain = float(
            summary["realized_genetic_gain"]
        )
        self._previous_variance_retention = (
            variance_retention
        )

        completed_generation = int(
            summary["generation"]
        )

        self._episode_return += terminal_reward
        self._episode_steps += 1

        terminated = (
            completed_generation
            >= self.config.maximum_generations
        )
        self._terminated = terminated

        if not terminated:
            self.bridge.start_generation()
            self._marker_matrix = (
                self.bridge.get_marker_matrix(
                    snp_chip=self.config.snp_chip
                )
            )
            self._prediction_table = None
            self._uncertainty_table = None
            self._previous_mean_reliability = 0.0
            self._last_uncertainty_error = None

        observation = self._build_current_observation()

        info = self._build_info(
            event="generation_finalized",
            reward_breakdown=complete_breakdown,
            selected_indices=selected_indices,
            action=action,
        )
        info["cycle_summary"] = (
            final_result["cycle_summary"].copy()
        )
        info["completed_generation"] = (
            completed_generation
        )
        info["variance_retention"] = (
            variance_retention
        )

        self._last_info = info

        return observation, float(reward), terminated, False, info

    def _build_info(
        self,
        *,
        event: str,
        reward_breakdown: RewardBreakdown | None,
        selected_indices: np.ndarray | None = None,
        action: PhenotypingAction | None = None,
    ) -> dict[str, Any]:
        """Build the standard Gymnasium info dictionary."""
        info: dict[str, Any] = {
            "event": event,
            "generation": self.bridge.generation,
            "number_phenotyped": (
                self.bridge.number_currently_phenotyped
            ),
            "action_mask": self.action_masks(),
            "episode_return": self._episode_return,
            "episode_steps": self._episode_steps,
            "uncertainty_available": (
                self._uncertainty_table is not None
            ),
        }

        if self._last_uncertainty_error is not None:
            info["uncertainty_error"] = (
                self._last_uncertainty_error
            )

        if action is not None:
            info["action"] = int(action)
            info["action_name"] = action_name(action)

        if selected_indices is not None:
            info["selected_indices"] = (
                np.asarray(
                    selected_indices,
                    dtype=np.int64,
                ).copy()
            )

        if reward_breakdown is not None:
            info["reward_breakdown"] = {
                "total": reward_breakdown.total,
                "genetic_gain_component": (
                    reward_breakdown.genetic_gain_component
                ),
                "variance_component": (
                    reward_breakdown.variance_component
                ),
                "cost_component": (
                    reward_breakdown.cost_component
                ),
                "model_quality_component": (
                    reward_breakdown.model_quality_component
                ),
                "invalid_action_component": (
                    reward_breakdown.invalid_action_component
                ),
            }

        return info

    def render(self) -> str:
        """Return a compact text description of the environment state."""
        mask = self.action_masks()
        valid_actions = [
            ACTION.name
            for ACTION in PhenotypingAction
            if mask[int(ACTION)]
        ]

        display_generation = min(
            self.bridge.generation,
            self.config.maximum_generations,
        )

        return (
            f"Generation {display_generation}/"
            f"{self.config.maximum_generations}; "
            f"phenotyped "
            f"{self.bridge.number_currently_phenotyped}/"
            f"{self.config.maximum_phenotypes}; "
            f"valid actions={valid_actions}"
        )
