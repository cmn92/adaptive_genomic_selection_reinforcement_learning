from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import pandas as pd

from src.environment.reward import RewardConfig
from src.evaluation.compare_strategies import StrategyComparisonConfig, compare_strategies
from src.evaluation.metrics import summarize_all_replicates
from src.evaluation.run_rl_strategy import RLStrategyEvaluationConfig, evaluate_frozen_rl_strategy
from src.rl.discretizer import compact_breeding_discretizer
from src.rl.linear_q import LinearQAgent
from src.rl.q_learning import load_q_agent


@dataclass
class FiveStrategyComparisonResult:
    generation_results: pd.DataFrame
    replicate_results: pd.DataFrame
    rl_step_history: pd.DataFrame


def compare_all_five_strategies(
    *,
    project_root: str | Path,
    agent_path: str | Path,
    number_of_replicates: int = 20,
    number_of_generations: int = 20,
    number_to_phenotype: int = 200,
    base_seed: int = 20001,
) -> FiveStrategyComparisonResult:
    project_root = Path(project_root).expanduser().resolve()
    agent = load_q_agent(agent_path)
    discretizer = (
        None
        if isinstance(agent, LinearQAgent)
        else compact_breeding_discretizer()
    )

    baseline = compare_strategies(
        project_root=project_root,
        config=StrategyComparisonConfig(
            number_of_replicates=number_of_replicates,
            number_of_generations=number_of_generations,
            number_to_phenotype=number_to_phenotype,
            number_of_parents=20,
            number_of_crosses=100,
            f1_per_cross=1,
            dh_per_f1=10,
            reps=1,
            trait=1,
            snp_chip=1,
            active_initial_batch_size=50,
            n_cores=1,
            base_seed=base_seed,
        ),
    )

    rl = evaluate_frozen_rl_strategy(
        project_root=project_root,
        agent=agent,
        discretizer=discretizer,
        config=RLStrategyEvaluationConfig(
            number_of_replicates=number_of_replicates,
            number_of_generations=number_of_generations,
            batch_size=25,
            minimum_training_size=50,
            maximum_phenotypes=number_to_phenotype,
            base_seed=base_seed,
        ),
        reward_config=RewardConfig(
            # The baseline strategies all use a fixed phenotyping budget.
            # Disable cost pressure here so RL is compared on batch choice,
            # not on its ability to spend less than the baselines.
            phenotyping_cost_weight=0.0,
        ),
    )

    generation_results = pd.concat(
        [baseline.generation_results, rl.generation_results],
        ignore_index=True,
        sort=False,
    ).sort_values(
        ["replicate", "strategy", "generation"]
    ).reset_index(drop=True)

    return FiveStrategyComparisonResult(
        generation_results=generation_results,
        replicate_results=summarize_all_replicates(generation_results),
        rl_step_history=rl.step_history,
    )


def save_five_strategy_results(result, output_directory: str | Path):
    output_directory = Path(output_directory).expanduser().resolve()
    raw = output_directory / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    paths = {
        "generation_results": raw / "generation_results.csv",
        "replicate_results": raw / "replicate_results.csv",
        "rl_step_history": raw / "rl_step_history.csv",
    }
    result.generation_results.to_csv(paths["generation_results"], index=False)
    result.replicate_results.to_csv(paths["replicate_results"], index=False)
    result.rl_step_history.to_csv(paths["rl_step_history"], index=False)
    return paths
