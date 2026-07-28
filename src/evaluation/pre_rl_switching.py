"""
pre_rl_switching.py

Pre-RL experiments for deciding whether a switching policy is worth training.

Phase 1 asks whether the best fixed phenotyping strategy changes across
generations or breeding conditions.

Phase 2 estimates the value of switching by comparing the best fixed strategy
against one practical switching policy. These policies are empirical decision
rules, not globally optimal oracles.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from src.baselines.active_learning import ActiveLearningStrategy
from src.baselines.base_strategy import BasePhenotypingStrategy
from src.baselines.diversity_sampling import DiversitySamplingStrategy
from src.baselines.fixed_sampling import FixedSamplingStrategy
from src.baselines.model_assisted import HighestGEBVStrategy
from src.baselines.random_sampling import RandomSamplingStrategy
from src.environment.r_bridge import RBreedingBridge
from src.evaluation.metrics import (
    summarize_strategy_replicate,
)


@dataclass(frozen=True)
class SwitchingScenario:
    """One breeding condition to evaluate before RL training."""

    name: str
    heritability: float | None = None
    population_size: int | None = None
    diversity_loss: str = "standard"
    number_of_replicates: int = 3
    number_of_generations: int = 8
    number_to_phenotype: int = 200
    number_of_parents: int = 20
    number_of_crosses: int = 100
    f1_per_cross: int = 1
    dh_per_f1: int = 10
    reps: int = 1
    trait: int = 1
    snp_chip: int = 1
    n_cores: int = 1
    active_initial_batch_size: int = 50
    base_seed: int = 30001

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("'name' must be a non-empty string.")

        if self.diversity_loss not in {
            "weak",
            "standard",
            "strong",
        }:
            raise ValueError(
                "'diversity_loss' must be 'weak', 'standard', or "
                "'strong'."
            )

        integer_fields = {
            "number_of_replicates": self.number_of_replicates,
            "number_of_generations": self.number_of_generations,
            "number_to_phenotype": self.number_to_phenotype,
            "number_of_parents": self.number_of_parents,
            "number_of_crosses": self.number_of_crosses,
            "f1_per_cross": self.f1_per_cross,
            "dh_per_f1": self.dh_per_f1,
            "reps": self.reps,
            "trait": self.trait,
            "snp_chip": self.snp_chip,
            "n_cores": self.n_cores,
            "active_initial_batch_size": self.active_initial_batch_size,
            "base_seed": self.base_seed,
        }

        if self.population_size is not None:
            integer_fields["population_size"] = self.population_size

        for field_name, value in integer_fields.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or value < 1
            ):
                raise ValueError(
                    f"'{field_name}' must be a positive integer."
                )

        if self.number_of_parents > self.number_to_phenotype:
            raise ValueError(
                "'number_of_parents' cannot exceed "
                "'number_to_phenotype'."
            )

        if (
            self.population_size is not None
            and self.number_to_phenotype > self.population_size
        ):
            raise ValueError(
                "'number_to_phenotype' cannot exceed "
                "'population_size'."
            )

        if (
            self.population_size is not None
            and self.number_of_parents > self.population_size
        ):
            raise ValueError(
                "'number_of_parents' cannot exceed 'population_size'."
            )

        expected_population = (
            self.number_of_crosses
            * self.f1_per_cross
            * self.dh_per_f1
        )

        if (
            self.population_size is not None
            and expected_population != self.population_size
        ):
            raise ValueError(
                "For population-size scenarios, number_of_crosses * "
                "f1_per_cross * dh_per_f1 must equal population_size."
            )

        if self.active_initial_batch_size >= self.number_to_phenotype:
            raise ValueError(
                "'active_initial_batch_size' must be smaller than "
                "'number_to_phenotype'."
            )

        if self.heritability is not None:
            numeric = float(self.heritability)
            if not np.isfinite(numeric) or not 0.0 < numeric <= 1.0:
                raise ValueError(
                    "'heritability' must be greater than zero and at "
                    "most one."
                )


@dataclass(frozen=True)
class SwitchingPolicyConfig:
    """Configuration for the empirical switching policy."""

    policy: str = "one_step_greedy"
    action_repeats: int = 1
    rollout_continuation: str = "best_fixed"

    def __post_init__(self) -> None:
        if self.policy not in {
            "one_step_greedy",
            "averaged_one_step",
            "rollout_to_end",
        }:
            raise ValueError(
                "'policy' must be 'one_step_greedy', "
                "'averaged_one_step', or 'rollout_to_end'."
            )

        if (
            isinstance(self.action_repeats, bool)
            or not isinstance(self.action_repeats, (int, np.integer))
            or self.action_repeats < 1
        ):
            raise ValueError(
                "'action_repeats' must be a positive integer."
            )

        if self.rollout_continuation != "best_fixed":
            raise ValueError(
                "Only rollout_continuation='best_fixed' is currently "
                "implemented."
            )

    @property
    def strategy_name(self) -> str:
        """Return the strategy label used in output tables."""
        if self.policy == "one_step_greedy":
            return "one_step_greedy_switching"

        if self.policy == "averaged_one_step":
            return (
                f"averaged_one_step_switching_"
                f"{self.action_repeats}_draws"
            )

        return "rollout_to_end_switching"


@dataclass
class SwitchingAnalysisResult:
    """Outputs from the pre-RL switching-value analysis."""

    scenarios: list[SwitchingScenario]
    switching_policy: SwitchingPolicyConfig
    fixed_generation_results: pd.DataFrame
    fixed_replicate_results: pd.DataFrame
    generation_winners: pd.DataFrame
    switching_summary: pd.DataFrame
    switching_generation_results: pd.DataFrame
    switching_replicate_results: pd.DataFrame
    switching_trial_results: pd.DataFrame
    switching_advantage: pd.DataFrame
    total_runtime_seconds: float


def default_switching_scenarios() -> list[SwitchingScenario]:
    """Return a small default grid for pre-RL development runs."""
    return [
        SwitchingScenario(
            name="h2_0.05_budget_200_pop_1000_parents_20_gen_8_loss_standard",
            heritability=0.05,
            population_size=1000,
            number_of_crosses=100,
            dh_per_f1=10,
        ),
        SwitchingScenario(
            name="h2_0.10_budget_200_pop_1000_parents_20_gen_8_loss_standard",
            heritability=0.10,
            population_size=1000,
            number_of_crosses=100,
            dh_per_f1=10,
            base_seed=31001,
        ),
        SwitchingScenario(
            name="h2_0.40_budget_200_pop_1000_parents_20_gen_8_loss_standard",
            heritability=0.40,
            base_seed=32001,
            population_size=1000,
            number_of_crosses=100,
            dh_per_f1=10,
        ),
    ]


def _strategy_set(
    scenario: SwitchingScenario,
) -> dict[str, BasePhenotypingStrategy | ActiveLearningStrategy]:
    """Create fresh strategy objects for one scenario."""
    return {
        "random_sampling": RandomSamplingStrategy(
            sort_indices=True
        ),
        "fixed_sampling": FixedSamplingStrategy(
            selection_rule="evenly_spaced",
            sort_indices=True,
        ),
        "diversity_sampling": DiversitySamplingStrategy(
            initial_selection="centroid_farthest",
            standardize_markers=True,
            sort_indices=True,
        ),
        "active_learning_pev": ActiveLearningStrategy(
            initial_batch_size=scenario.active_initial_batch_size,
            name="active_learning_pev",
        ),
        "highest_gebv": HighestGEBVStrategy(
            initial_batch_size=scenario.active_initial_batch_size,
        ),
    }


def _prepare_bridge(
    *,
    project_root: Path,
    population_file: str | Path,
    seed: int,
    scenario: SwitchingScenario,
) -> RBreedingBridge:
    """Create and reset a bridge for one scenario replicate."""
    bridge = RBreedingBridge(
        project_root=project_root,
        population_file=population_file,
        seed=seed,
    )
    bridge.reset(seed=seed)

    if scenario.heritability is not None:
        bridge.set_trait_heritability(
            scenario.heritability
        )

    if scenario.population_size is not None:
        bridge.subset_current_population(
            population_size=scenario.population_size,
            seed=seed,
        )

    return bridge


def _run_one_generation(
    *,
    bridge: RBreedingBridge,
    strategy: BasePhenotypingStrategy | ActiveLearningStrategy,
    scenario: SwitchingScenario,
    rng: np.random.Generator,
    seed: int,
) -> pd.DataFrame:
    """Run one strategy for the current bridge generation."""
    generation = int(bridge.generation)

    if isinstance(strategy, ActiveLearningStrategy):
        generation_result = strategy.run_generation(
            bridge=bridge,
            number_to_phenotype=scenario.number_to_phenotype,
            rng=rng,
            number_of_parents=scenario.number_of_parents,
            number_of_crosses=scenario.number_of_crosses,
            f1_per_cross=scenario.f1_per_cross,
            dh_per_f1=scenario.dh_per_f1,
            reps=scenario.reps,
            trait=scenario.trait,
            snp_chip=scenario.snp_chip,
            n_cores=scenario.n_cores,
            seed=seed,
        )
        summary = generation_result.cycle_summary.copy()
    else:
        include_markers = isinstance(
            strategy,
            DiversitySamplingStrategy,
        )
        candidate_data = bridge.get_candidate_data(
            include_markers=include_markers
        )
        selected_indices = strategy.select(
            candidate_data=candidate_data,
            number_to_phenotype=scenario.number_to_phenotype,
            rng=rng,
        )
        cycle_result = bridge.step(
            selected_indices=selected_indices,
            number_of_parents=scenario.number_of_parents,
            number_of_crosses=scenario.number_of_crosses,
            f1_per_cross=scenario.f1_per_cross,
            dh_per_f1=scenario.dh_per_f1,
            reps=scenario.reps,
            trait=scenario.trait,
            seed=seed,
        )
        summary = cycle_result["cycle_summary"].copy()
        summary.insert(0, "strategy", strategy.name)
        summary["selection_seconds"] = np.nan
        summary["cycle_seconds"] = np.nan

    if int(summary.loc[0, "generation"]) != generation:
        raise RuntimeError(
            "Generation summary does not match the bridge state."
        )

    return summary


def _add_context(
    table: pd.DataFrame,
    *,
    scenario: SwitchingScenario,
    replicate: int,
    seed: int,
) -> pd.DataFrame:
    """Add scenario and replicate metadata to one generation table."""
    result = table.copy()
    result.insert(0, "scenario", scenario.name)

    if "replicate" not in result.columns:
        result.insert(2, "replicate", int(replicate))
    else:
        result["replicate"] = int(replicate)

    result["run_seed"] = int(seed)
    result["scenario_heritability"] = (
        np.nan
        if scenario.heritability is None
        else float(scenario.heritability)
    )
    result["scenario_number_to_phenotype"] = (
        scenario.number_to_phenotype
    )
    result["scenario_population_size"] = scenario.population_size
    result["scenario_diversity_loss"] = scenario.diversity_loss
    result["scenario_number_of_parents"] = (
        scenario.number_of_parents
    )
    result["scenario_number_of_crosses"] = (
        scenario.number_of_crosses
    )
    result["scenario_dh_per_f1"] = scenario.dh_per_f1
    result["scenario_number_of_generations"] = (
        scenario.number_of_generations
    )

    return result


def _summarize_by_scenario(
    generation_results: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize strategy-replicate trajectories within each scenario."""
    rows: list[pd.DataFrame] = []

    for (scenario, strategy, replicate), group in generation_results.groupby(
        ["scenario", "strategy", "replicate"],
        sort=True,
    ):
        summary = summarize_strategy_replicate(
            group,
            replicate=int(replicate),
            seed=int(group["run_seed"].iloc[0]),
        )
        summary.insert(0, "scenario", scenario)
        rows.append(summary)

    return pd.concat(rows, ignore_index=True)


def run_fixed_strategy_panel(
    *,
    project_root: str | Path,
    scenarios: list[SwitchingScenario],
    population_file: str | Path = "data/initial_candidate_population.RData",
) -> pd.DataFrame:
    """Run all fixed strategies for each scenario and replicate."""
    project_root = Path(project_root).expanduser().resolve()
    generation_tables: list[pd.DataFrame] = []

    for scenario in scenarios:
        strategies = _strategy_set(scenario)

        for replicate in range(1, scenario.number_of_replicates + 1):
            seed = scenario.base_seed + replicate - 1

            for strategy_name, strategy in strategies.items():
                print(
                    f"\n[Phase 1] {scenario.name} | replicate "
                    f"{replicate}/{scenario.number_of_replicates} | "
                    f"{strategy_name}"
                )
                bridge = _prepare_bridge(
                    project_root=project_root,
                    population_file=population_file,
                    seed=seed,
                    scenario=scenario,
                )
                rng = np.random.default_rng(seed)

                for generation in range(
                    1,
                    scenario.number_of_generations + 1,
                ):
                    generation_seed = seed + generation - 1
                    summary = _run_one_generation(
                        bridge=bridge,
                        strategy=strategy,
                        scenario=scenario,
                        rng=rng,
                        seed=generation_seed,
                    )
                    generation_tables.append(
                        _add_context(
                            summary,
                            scenario=scenario,
                            replicate=replicate,
                            seed=seed,
                        )
                    )

    return pd.concat(
        generation_tables,
        ignore_index=True,
        sort=False,
    )


def calculate_generation_winners(
    generation_results: pd.DataFrame,
    *,
    metric: str = "realized_genetic_gain",
) -> pd.DataFrame:
    """Return the best fixed strategy by scenario and generation."""
    grouped = (
        generation_results.groupby(
            ["scenario", "generation", "strategy"],
            as_index=False,
        )
        .agg(
            mean_metric=(metric, "mean"),
            standard_deviation=(metric, "std"),
            count=(metric, "count"),
        )
    )
    grouped["standard_error"] = (
        grouped["standard_deviation"]
        / np.sqrt(grouped["count"])
    )
    grouped["standard_error"] = grouped[
        "standard_error"
    ].fillna(0.0)

    grouped = grouped.sort_values(
        ["scenario", "generation", "mean_metric", "strategy"],
        ascending=[True, True, False, True],
    )

    rows: list[dict[str, Any]] = []

    for (scenario, generation), group in grouped.groupby(
        ["scenario", "generation"],
        sort=True,
    ):
        ordered = group.sort_values(
            ["mean_metric", "strategy"],
            ascending=[False, True],
        ).reset_index(drop=True)
        winner = ordered.iloc[0]
        runner_up = (
            ordered.iloc[1]
            if len(ordered) > 1
            else None
        )
        runner_up_strategy = (
            str(runner_up["strategy"])
            if runner_up is not None
            else ""
        )
        runner_up_mean = (
            float(runner_up["mean_metric"])
            if runner_up is not None
            else np.nan
        )

        rows.append(
            {
                "scenario": scenario,
                "generation": int(generation),
                "best_strategy": str(winner["strategy"]),
                f"best_mean_{metric}": float(
                    winner["mean_metric"]
                ),
                "best_standard_error": float(
                    winner["standard_error"]
                ),
                "winner_replicates": int(winner["count"]),
                "runner_up_strategy": runner_up_strategy,
                f"runner_up_mean_{metric}": runner_up_mean,
                "winner_margin": (
                    float(winner["mean_metric"])
                    - runner_up_mean
                    if runner_up is not None
                    else np.nan
                ),
                "low_replication_warning": int(
                    winner["count"]
                )
                < 20,
            }
        )

    return pd.DataFrame(rows)


def summarize_switching_patterns(
    generation_winners: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize whether winners change over time within each scenario."""
    rows: list[dict[str, Any]] = []

    for scenario, group in generation_winners.groupby(
        "scenario",
        sort=True,
    ):
        ordered = group.sort_values("generation")
        winners = ordered["best_strategy"].astype(str).tolist()
        transitions = sum(
            previous != current
            for previous, current in zip(
                winners[:-1],
                winners[1:],
            )
        )
        rows.append(
            {
                "scenario": scenario,
                "number_of_generations": len(winners),
                "number_of_unique_winners": len(set(winners)),
                "number_of_strategy_switches": transitions,
                "switching_detected": transitions > 0,
                "minimum_winner_margin": float(
                    ordered["winner_margin"].min()
                ),
                "median_winner_margin": float(
                    ordered["winner_margin"].median()
                ),
                "minimum_winner_replicates": int(
                    ordered["winner_replicates"].min()
                ),
                "low_replication_warning": bool(
                    ordered["low_replication_warning"].any()
                ),
                "winner_sequence": " -> ".join(winners),
            }
        )

    return pd.DataFrame(rows)


def best_fixed_strategy_by_scenario(
    fixed_replicate_results: pd.DataFrame,
) -> dict[str, str]:
    """Return the best fixed strategy name for each scenario."""
    means = (
        fixed_replicate_results.groupby(
            ["scenario", "strategy"],
            as_index=False,
        )["total_realized_genetic_gain"]
        .mean()
        .rename(
            columns={
                "total_realized_genetic_gain": "mean_total_gain"
            }
        )
    )

    winners: dict[str, str] = {}

    for scenario, group in means.groupby("scenario", sort=True):
        winner = group.sort_values(
            ["mean_total_gain", "strategy"],
            ascending=[False, True],
        ).iloc[0]
        winners[str(scenario)] = str(winner["strategy"])

    return winners


def _trajectory_total_gain(
    rows: list[pd.DataFrame],
) -> float:
    """Return total gain over a temporary trajectory."""
    trajectory = pd.concat(
        rows,
        ignore_index=True,
        sort=False,
    )
    first = trajectory.iloc[0]
    last = trajectory.iloc[-1]
    return float(
        last["next_generation_mean_gv"]
        - first["population_mean_gv_before"]
    )


def _score_candidate_strategy(
    *,
    bridge: RBreedingBridge,
    state_path: Path,
    strategy: BasePhenotypingStrategy | ActiveLearningStrategy,
    scenario: SwitchingScenario,
    generation: int,
    run_seed: int,
    strategy_seed: int,
    policy_config: SwitchingPolicyConfig,
    continuation_strategy: BasePhenotypingStrategy | ActiveLearningStrategy,
) -> tuple[pd.DataFrame, float]:
    """Score one first-generation candidate strategy."""
    bridge.load_program_state(state_path)
    first_summary = _run_one_generation(
        bridge=bridge,
        strategy=strategy,
        scenario=scenario,
        rng=np.random.default_rng(strategy_seed),
        seed=run_seed + generation - 1,
    )

    if policy_config.policy != "rollout_to_end":
        return first_summary, float(
            first_summary.loc[0, "realized_genetic_gain"]
        )

    rollout_rows = [first_summary]

    for future_generation in range(
        generation + 1,
        scenario.number_of_generations + 1,
    ):
        continuation_seed = (
            strategy_seed
            + future_generation * 1000
        )
        continuation_summary = _run_one_generation(
            bridge=bridge,
            strategy=continuation_strategy,
            scenario=scenario,
            rng=np.random.default_rng(continuation_seed),
            seed=run_seed + future_generation - 1,
        )
        rollout_rows.append(continuation_summary)

    return first_summary, _trajectory_total_gain(rollout_rows)


def run_switching_policy(
    *,
    project_root: str | Path,
    scenarios: list[SwitchingScenario],
    policy_config: SwitchingPolicyConfig,
    best_fixed_by_scenario: dict[str, str],
    population_file: str | Path = "data/initial_candidate_population.RData",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run a same-state empirical switching policy for each replicate.

    Returns
    -------
    tuple
        Chosen switching generation rows and all trial rows.
    """
    project_root = Path(project_root).expanduser().resolve()
    switching_rows: list[pd.DataFrame] = []
    trial_rows: list[pd.DataFrame] = []

    with TemporaryDirectory() as temporary:
        temporary_root = Path(temporary)

        for scenario in scenarios:
            strategies = _strategy_set(scenario)

            for replicate in range(1, scenario.number_of_replicates + 1):
                seed = scenario.base_seed + replicate - 1
                bridge = _prepare_bridge(
                    project_root=project_root,
                    population_file=population_file,
                    seed=seed,
                    scenario=scenario,
                )

                for generation in range(
                    1,
                    scenario.number_of_generations + 1,
                ):
                    print(
                        f"\n[Phase 2] {scenario.name} | replicate "
                        f"{replicate}/{scenario.number_of_replicates} | "
                        f"{policy_config.strategy_name} generation "
                        f"{generation}"
                    )

                    state_path = temporary_root / (
                        f"{scenario.name}_rep{replicate}_"
                        f"gen{generation}.RData"
                    )
                    bridge.save_program_state(state_path)

                    candidate_rows: list[pd.DataFrame] = []
                    strategy_seed_by_name: dict[str, int] = {}
                    strategy_score_by_name: dict[str, float] = {}

                    for strategy_index, (
                        strategy_name,
                        strategy,
                    ) in enumerate(strategies.items()):
                        repeat_scores: list[float] = []
                        repeat_count = (
                            1
                            if policy_config.policy == "one_step_greedy"
                            else policy_config.action_repeats
                        )

                        for repeat in range(
                            repeat_count
                        ):
                            strategy_seed = (
                                seed
                                + generation * 1000
                                + strategy_index * 100
                                + repeat
                            )
                            strategy_seed_by_name.setdefault(
                                strategy_name,
                                strategy_seed,
                            )
                            continuation_name = (
                                best_fixed_by_scenario[scenario.name]
                            )
                            summary, score = _score_candidate_strategy(
                                bridge=bridge,
                                state_path=state_path,
                                strategy=strategy,
                                scenario=scenario,
                                generation=generation,
                                run_seed=seed,
                                strategy_seed=strategy_seed,
                                policy_config=policy_config,
                                continuation_strategy=strategies[
                                    continuation_name
                                ],
                            )
                            repeat_scores.append(score)
                            summary = _add_context(
                                summary,
                                scenario=scenario,
                                replicate=replicate,
                                seed=seed,
                            )
                            summary["switching_trial_strategy"] = (
                                strategy_name
                            )
                            summary["switching_trial_repeat"] = (
                                repeat + 1
                            )
                            summary["switching_trial_score"] = score
                            summary["switching_policy"] = (
                                policy_config.strategy_name
                            )
                            summary["rollout_continuation"] = (
                                policy_config.rollout_continuation
                                if policy_config.policy == "rollout_to_end"
                                else ""
                            )
                            candidate_rows.append(summary)

                        strategy_score_by_name[strategy_name] = float(
                            np.mean(repeat_scores)
                        )

                    trial_table = pd.concat(
                        candidate_rows,
                        ignore_index=True,
                        sort=False,
                    )
                    trial_rows.append(trial_table)

                    best_strategy = max(
                        strategy_score_by_name,
                        key=lambda name: (
                            strategy_score_by_name[name],
                            name,
                        ),
                    )

                    bridge.load_program_state(state_path)
                    committed_summary = _run_one_generation(
                        bridge=bridge,
                        strategy=strategies[best_strategy],
                        scenario=scenario,
                        rng=np.random.default_rng(
                            strategy_seed_by_name[best_strategy]
                        ),
                        seed=seed + generation - 1,
                    )
                    committed_summary = _add_context(
                        committed_summary,
                        scenario=scenario,
                        replicate=replicate,
                        seed=seed,
                    )
                    committed_summary["strategy"] = (
                        policy_config.strategy_name
                    )
                    committed_summary["switching_chosen_strategy"] = (
                        best_strategy
                    )
                    committed_summary["switching_policy"] = (
                        policy_config.strategy_name
                    )
                    committed_summary["chosen_strategy_score"] = (
                        strategy_score_by_name[best_strategy]
                    )
                    committed_summary["best_fixed_continuation"] = (
                        best_fixed_by_scenario[scenario.name]
                    )
                    switching_rows.append(committed_summary)

    switching_generation_results = pd.concat(
        switching_rows,
        ignore_index=True,
        sort=False,
    )
    switching_trial_results = pd.concat(
        trial_rows,
        ignore_index=True,
        sort=False,
    )

    return switching_generation_results, switching_trial_results


def calculate_switching_advantage(
    *,
    fixed_replicate_results: pd.DataFrame,
    switching_replicate_results: pd.DataFrame,
) -> pd.DataFrame:
    """Compare switching against best fixed and report a safe ceiling."""
    rows: list[dict[str, Any]] = []

    fixed_means = (
        fixed_replicate_results.groupby(
            ["scenario", "strategy"],
            as_index=False,
        )["total_realized_genetic_gain"]
        .mean()
        .rename(
            columns={
                "total_realized_genetic_gain": "mean_total_gain"
            }
        )
    )

    switching_means = (
        switching_replicate_results.groupby(
            ["scenario", "strategy"],
            as_index=False,
        )["total_realized_genetic_gain"]
        .mean()
        .rename(
            columns={
                "total_realized_genetic_gain": (
                    "switching_mean_total_gain"
                )
            }
        )
    )

    for scenario, group in fixed_means.groupby("scenario", sort=True):
        best_fixed = group.sort_values(
            ["mean_total_gain", "strategy"],
            ascending=[False, True],
        ).iloc[0]
        switching_row = switching_means[
            switching_means["scenario"] == scenario
        ].iloc[0]
        switching_gain = float(
            switching_row["switching_mean_total_gain"]
        )
        best_fixed_gain = float(best_fixed["mean_total_gain"])
        ceiling_gain = max(switching_gain, best_fixed_gain)

        rows.append(
            {
                "scenario": scenario,
                "switching_policy": switching_row["strategy"],
                "best_fixed_strategy": best_fixed["strategy"],
                "best_fixed_mean_total_gain": best_fixed_gain,
                "switching_mean_total_gain": switching_gain,
                "raw_switching_advantage": float(
                    switching_gain - best_fixed_gain
                ),
                "ceiling_mean_total_gain": ceiling_gain,
                "ceiling_advantage": float(
                    ceiling_gain - best_fixed_gain
                ),
                "ceiling_source": (
                    "switching_policy"
                    if switching_gain >= best_fixed_gain
                    else "best_fixed_floor"
                ),
            }
        )

    return pd.DataFrame(rows)


def run_pre_rl_switching_analysis(
    *,
    project_root: str | Path,
    scenarios: list[SwitchingScenario] | None = None,
    switching_policy: SwitchingPolicyConfig | None = None,
    population_file: str | Path = "data/initial_candidate_population.RData",
) -> SwitchingAnalysisResult:
    """Run Phase 1 and Phase 2 pre-RL switching analysis."""
    scenarios = scenarios or default_switching_scenarios()
    switching_policy = switching_policy or SwitchingPolicyConfig()
    start = perf_counter()

    fixed_generation_results = run_fixed_strategy_panel(
        project_root=project_root,
        scenarios=scenarios,
        population_file=population_file,
    )
    fixed_replicate_results = _summarize_by_scenario(
        fixed_generation_results
    )
    generation_winners = calculate_generation_winners(
        fixed_generation_results
    )
    switching_summary = summarize_switching_patterns(
        generation_winners
    )

    best_fixed_by_scenario = best_fixed_strategy_by_scenario(
        fixed_replicate_results
    )
    switching_generation_results, switching_trial_results = (
        run_switching_policy(
            project_root=project_root,
            scenarios=scenarios,
            policy_config=switching_policy,
            best_fixed_by_scenario=best_fixed_by_scenario,
            population_file=population_file,
        )
    )
    switching_replicate_results = _summarize_by_scenario(
        switching_generation_results
    )
    switching_advantage = calculate_switching_advantage(
        fixed_replicate_results=fixed_replicate_results,
        switching_replicate_results=switching_replicate_results,
    )

    return SwitchingAnalysisResult(
        scenarios=scenarios,
        switching_policy=switching_policy,
        fixed_generation_results=fixed_generation_results,
        fixed_replicate_results=fixed_replicate_results,
        generation_winners=generation_winners,
        switching_summary=switching_summary,
        switching_generation_results=switching_generation_results,
        switching_replicate_results=switching_replicate_results,
        switching_trial_results=switching_trial_results,
        switching_advantage=switching_advantage,
        total_runtime_seconds=perf_counter() - start,
    )


def save_pre_rl_switching_analysis(
    result: SwitchingAnalysisResult,
    output_directory: str | Path,
) -> dict[str, Path]:
    """Save all pre-RL switching analysis outputs."""
    output_directory = Path(output_directory).expanduser().resolve()
    raw = output_directory / "raw"
    processed = output_directory / "processed"
    raw.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)

    paths = {
        "fixed_generation_results": (
            raw / "fixed_generation_results.csv"
        ),
        "fixed_replicate_results": (
            raw / "fixed_replicate_results.csv"
        ),
        "switching_generation_results": (
            raw / "switching_generation_results.csv"
        ),
        "switching_trial_results": (
            raw / "switching_trial_results.csv"
        ),
        "generation_winners": (
            processed / "generation_winners.csv"
        ),
        "switching_summary": (
            processed / "switching_summary.csv"
        ),
        "switching_replicate_results": (
            processed / "switching_replicate_results.csv"
        ),
        "switching_advantage": (
            processed / "switching_advantage.csv"
        ),
        "scenario_configuration": (
            raw / "scenario_configuration.csv"
        ),
        "switching_policy_configuration": (
            raw / "switching_policy_configuration.csv"
        ),
        "report": output_directory / "pre_rl_switching_report.md",
    }

    result.fixed_generation_results.to_csv(
        paths["fixed_generation_results"],
        index=False,
    )
    result.fixed_replicate_results.to_csv(
        paths["fixed_replicate_results"],
        index=False,
    )
    result.switching_generation_results.to_csv(
        paths["switching_generation_results"],
        index=False,
    )
    result.switching_trial_results.to_csv(
        paths["switching_trial_results"],
        index=False,
    )
    result.generation_winners.to_csv(
        paths["generation_winners"],
        index=False,
    )
    result.switching_summary.to_csv(
        paths["switching_summary"],
        index=False,
    )
    result.switching_replicate_results.to_csv(
        paths["switching_replicate_results"],
        index=False,
    )
    result.switching_advantage.to_csv(
        paths["switching_advantage"],
        index=False,
    )
    pd.DataFrame(
        [scenario.__dict__ for scenario in result.scenarios]
    ).assign(
        total_runtime_seconds=result.total_runtime_seconds
    ).to_csv(
        paths["scenario_configuration"],
        index=False,
    )
    pd.DataFrame(
        [result.switching_policy.__dict__]
    ).assign(
        strategy_name=result.switching_policy.strategy_name
    ).to_csv(
        paths["switching_policy_configuration"],
        index=False,
    )

    paths["report"].write_text(
        _render_report(result),
        encoding="utf-8",
    )

    return paths


def _render_report(
    result: SwitchingAnalysisResult,
) -> str:
    """Render a compact markdown report."""
    lines = [
        "# Pre-RL Switching Analysis",
        "",
        "## Phase 1: Is Switching Useful?",
        "",
        _plain_table(result.switching_summary),
        "",
        "## Phase 2: Best Fixed Strategy vs Empirical Switching",
        "",
        _plain_table(result.switching_advantage),
        "",
        "## Interpretation",
        "",
        (
            "The switching policy is not a mathematically perfect oracle. "
            "Use raw_switching_advantage to see how the empirical policy "
            "performed, and ceiling_advantage to see the nonnegative upper "
            "candidate obtained by flooring at the best fixed strategy. "
            "Winner sequences with low_replication_warning=True should be "
            "treated as exploratory rather than conclusive."
        ),
        "",
        "Total runtime seconds: "
        f"{result.total_runtime_seconds:.3f}",
        "",
    ]

    return "\n".join(lines)


def _plain_table(table: pd.DataFrame) -> str:
    """Return a dependency-free markdown-friendly table block."""
    return "```text\n" + table.to_string(index=False) + "\n```"
