"""
pre_rl_switching.py

Pre-RL experiments for deciding whether a switching policy is worth training.

Phase 1 asks whether the best fixed phenotyping strategy changes across
generations or breeding conditions.

Phase 2 estimates the value of switching by comparing the best fixed strategy
against an oracle that branches from the same population state, tries each
available strategy for one generation, and commits the strategy with the
largest immediate realized genetic gain.
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


@dataclass
class SwitchingAnalysisResult:
    """Outputs from the pre-RL switching-value analysis."""

    scenarios: list[SwitchingScenario]
    fixed_generation_results: pd.DataFrame
    fixed_replicate_results: pd.DataFrame
    generation_winners: pd.DataFrame
    switching_summary: pd.DataFrame
    oracle_generation_results: pd.DataFrame
    oracle_replicate_results: pd.DataFrame
    oracle_trial_results: pd.DataFrame
    oracle_advantage: pd.DataFrame
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
        )[metric]
        .mean()
        .rename(columns={metric: f"mean_{metric}"})
    )

    grouped = grouped.sort_values(
        ["scenario", "generation", f"mean_{metric}", "strategy"],
        ascending=[True, True, False, True],
    )

    winners = (
        grouped.groupby(
            ["scenario", "generation"],
            as_index=False,
        )
        .head(1)
        .reset_index(drop=True)
    )
    winners = winners.rename(
        columns={"strategy": "best_strategy"}
    )

    return winners


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
                "winner_sequence": " -> ".join(winners),
            }
        )

    return pd.DataFrame(rows)


def run_oracle_switching(
    *,
    project_root: str | Path,
    scenarios: list[SwitchingScenario],
    population_file: str | Path = "data/initial_candidate_population.RData",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run a same-state one-generation oracle for each scenario replicate.

    Returns
    -------
    tuple
        Chosen oracle generation rows and all one-step trial rows.
    """
    project_root = Path(project_root).expanduser().resolve()
    oracle_rows: list[pd.DataFrame] = []
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
                        f"oracle generation {generation}"
                    )

                    state_path = temporary_root / (
                        f"{scenario.name}_rep{replicate}_"
                        f"gen{generation}.RData"
                    )
                    bridge.save_program_state(state_path)

                    candidate_rows: list[pd.DataFrame] = []
                    strategy_seeds: dict[str, int] = {}

                    for strategy_index, (
                        strategy_name,
                        strategy,
                    ) in enumerate(strategies.items()):
                        bridge.load_program_state(state_path)
                        strategy_seed = (
                            seed
                            + generation * 1000
                            + strategy_index
                        )
                        strategy_seeds[strategy_name] = strategy_seed
                        summary = _run_one_generation(
                            bridge=bridge,
                            strategy=strategy,
                            scenario=scenario,
                            rng=np.random.default_rng(strategy_seed),
                            seed=seed + generation - 1,
                        )
                        summary = _add_context(
                            summary,
                            scenario=scenario,
                            replicate=replicate,
                            seed=seed,
                        )
                        summary["oracle_trial_strategy"] = (
                            strategy_name
                        )
                        candidate_rows.append(summary)

                    trial_table = pd.concat(
                        candidate_rows,
                        ignore_index=True,
                        sort=False,
                    )
                    trial_rows.append(trial_table)

                    best_index = (
                        pd.to_numeric(
                            trial_table["realized_genetic_gain"],
                            errors="coerce",
                        )
                        .idxmax()
                    )
                    best_strategy = str(
                        trial_table.loc[
                            best_index,
                            "oracle_trial_strategy",
                        ]
                    )

                    bridge.load_program_state(state_path)
                    committed_summary = _run_one_generation(
                        bridge=bridge,
                        strategy=strategies[best_strategy],
                        scenario=scenario,
                        rng=np.random.default_rng(
                            strategy_seeds[best_strategy]
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
                        "oracle_switching"
                    )
                    committed_summary["oracle_chosen_strategy"] = (
                        best_strategy
                    )
                    oracle_rows.append(committed_summary)

    oracle_generation_results = pd.concat(
        oracle_rows,
        ignore_index=True,
        sort=False,
    )
    oracle_trial_results = pd.concat(
        trial_rows,
        ignore_index=True,
        sort=False,
    )

    return oracle_generation_results, oracle_trial_results


def calculate_oracle_advantage(
    *,
    fixed_replicate_results: pd.DataFrame,
    oracle_replicate_results: pd.DataFrame,
) -> pd.DataFrame:
    """Compare oracle switching against the best fixed strategy."""
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

    oracle_means = (
        oracle_replicate_results.groupby(
            "scenario",
            as_index=False,
        )["total_realized_genetic_gain"]
        .mean()
        .rename(
            columns={
                "total_realized_genetic_gain": (
                    "oracle_mean_total_gain"
                )
            }
        )
    )

    for scenario, group in fixed_means.groupby("scenario", sort=True):
        best_fixed = group.sort_values(
            ["mean_total_gain", "strategy"],
            ascending=[False, True],
        ).iloc[0]
        oracle_row = oracle_means[
            oracle_means["scenario"] == scenario
        ].iloc[0]

        rows.append(
            {
                "scenario": scenario,
                "best_fixed_strategy": best_fixed["strategy"],
                "best_fixed_mean_total_gain": float(
                    best_fixed["mean_total_gain"]
                ),
                "oracle_mean_total_gain": float(
                    oracle_row["oracle_mean_total_gain"]
                ),
                "oracle_gain_advantage": float(
                    oracle_row["oracle_mean_total_gain"]
                    - best_fixed["mean_total_gain"]
                ),
            }
        )

    return pd.DataFrame(rows)


def run_pre_rl_switching_analysis(
    *,
    project_root: str | Path,
    scenarios: list[SwitchingScenario] | None = None,
    population_file: str | Path = "data/initial_candidate_population.RData",
) -> SwitchingAnalysisResult:
    """Run Phase 1 and Phase 2 pre-RL switching analysis."""
    scenarios = scenarios or default_switching_scenarios()
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

    oracle_generation_results, oracle_trial_results = (
        run_oracle_switching(
            project_root=project_root,
            scenarios=scenarios,
            population_file=population_file,
        )
    )
    oracle_replicate_results = _summarize_by_scenario(
        oracle_generation_results
    )
    oracle_advantage = calculate_oracle_advantage(
        fixed_replicate_results=fixed_replicate_results,
        oracle_replicate_results=oracle_replicate_results,
    )

    return SwitchingAnalysisResult(
        scenarios=scenarios,
        fixed_generation_results=fixed_generation_results,
        fixed_replicate_results=fixed_replicate_results,
        generation_winners=generation_winners,
        switching_summary=switching_summary,
        oracle_generation_results=oracle_generation_results,
        oracle_replicate_results=oracle_replicate_results,
        oracle_trial_results=oracle_trial_results,
        oracle_advantage=oracle_advantage,
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
        "oracle_generation_results": (
            raw / "oracle_generation_results.csv"
        ),
        "oracle_trial_results": raw / "oracle_trial_results.csv",
        "generation_winners": (
            processed / "generation_winners.csv"
        ),
        "switching_summary": (
            processed / "switching_summary.csv"
        ),
        "oracle_replicate_results": (
            processed / "oracle_replicate_results.csv"
        ),
        "oracle_advantage": (
            processed / "oracle_advantage.csv"
        ),
        "scenario_configuration": (
            raw / "scenario_configuration.csv"
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
    result.oracle_generation_results.to_csv(
        paths["oracle_generation_results"],
        index=False,
    )
    result.oracle_trial_results.to_csv(
        paths["oracle_trial_results"],
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
    result.oracle_replicate_results.to_csv(
        paths["oracle_replicate_results"],
        index=False,
    )
    result.oracle_advantage.to_csv(
        paths["oracle_advantage"],
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
        "## Phase 2: Best Fixed Strategy vs Oracle Switching",
        "",
        _plain_table(result.oracle_advantage),
        "",
        "## Interpretation",
        "",
        (
            "If the winner sequence rarely changes, a fixed heuristic may be "
            "enough. If oracle switching has little advantage over the best "
            "fixed strategy, RL has little room to improve. If winner "
            "sequences change and oracle advantage is large, then RL has a "
            "meaningful switching problem to learn."
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
