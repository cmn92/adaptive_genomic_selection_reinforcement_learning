"""Strategy evaluation, statistics, plotting, and reporting utilities."""

from src.evaluation.compare_strategies import (
    StrategyComparisonConfig,
    StrategyComparisonResult,
    compare_strategies,
    save_comparison_results,
)
from src.evaluation.metrics import (
    area_under_gain_curve,
    rate_of_variance_loss,
    summarize_all_replicates,
    summarize_strategy_replicate,
)
from src.evaluation.plots import create_all_plots
from src.evaluation.report import (
    build_report_from_csv,
    create_evaluation_report,
)
from src.evaluation.run_active_strategy import (
    ActiveStrategyRunConfig,
    ActiveStrategyRunResult,
    run_active_strategy,
    save_active_strategy_run,
)
from src.evaluation.run_strategy import (
    StrategyRunConfig,
    StrategyRunResult,
    run_strategy,
    save_strategy_run,
)
from src.evaluation.statistics import (
    pairwise_wilcoxon_tests,
    run_statistical_analysis,
)

__all__ = [
    "StrategyComparisonConfig",
    "StrategyComparisonResult",
    "compare_strategies",
    "save_comparison_results",
    "area_under_gain_curve",
    "rate_of_variance_loss",
    "summarize_all_replicates",
    "summarize_strategy_replicate",
    "create_all_plots",
    "build_report_from_csv",
    "create_evaluation_report",
    "ActiveStrategyRunConfig",
    "ActiveStrategyRunResult",
    "run_active_strategy",
    "save_active_strategy_run",
    "StrategyRunConfig",
    "StrategyRunResult",
    "run_strategy",
    "save_strategy_run",
    "pairwise_wilcoxon_tests",
    "run_statistical_analysis",
]
