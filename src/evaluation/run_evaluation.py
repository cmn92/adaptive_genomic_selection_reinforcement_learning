"""
run_evaluation.py

One-command development evaluation.

Run from the project root:

    python src/evaluation/run_evaluation.py

The default settings run 20 matched replicates, 20 generations, and all four
baseline strategies.
"""

from pathlib import Path

from src.evaluation.compare_strategies import (
    StrategyComparisonConfig,
    compare_strategies,
    save_comparison_results,
)
from src.evaluation.report import create_evaluation_report


def main() -> None:
    """Run comparison, statistics, plots, and Markdown reporting."""
    project_root = Path(__file__).resolve().parents[2]
    output_directory = (
        project_root / "results" / "evaluation"
    )

    config = StrategyComparisonConfig(
        number_of_replicates=20,
        number_of_generations=20,
        number_to_phenotype=200,
        number_of_parents=20,
        number_of_crosses=100,
        f1_per_cross=1,
        dh_per_f1=10,
        reps=1,
        trait=1,
        snp_chip=1,
        active_initial_batch_size=50,
        n_cores=1,
        base_seed=1001,
    )

    comparison = compare_strategies(
        project_root=project_root,
        config=config,
    )

    raw_paths = save_comparison_results(
        comparison,
        output_directory,
    )

    report_paths = create_evaluation_report(
        generation_results=(
            comparison.generation_results
        ),
        replicate_results=(
            comparison.replicate_results
        ),
        output_directory=output_directory,
    )

    print("\nEvaluation completed.")
    print(
        "Total runtime seconds:",
        round(comparison.total_runtime_seconds, 3),
    )

    print("\nRaw outputs:")
    for name, path in raw_paths.items():
        print(f"{name}: {path}")

    print("\nReport outputs:")
    for name, path in report_paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
