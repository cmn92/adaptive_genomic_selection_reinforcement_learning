"""
report.py

Create a concise Markdown evaluation report from saved result tables,
statistical analyses, and figures.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.plots import create_all_plots
from src.evaluation.statistics import (
    DEFAULT_METRICS,
    run_statistical_analysis,
    save_statistical_results,
)


def _format_number(value: object, digits: int = 3) -> str:
    """Format numeric report values safely."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)

    if not np.isfinite(numeric):
        return "NA"

    return f"{numeric:.{digits}f}"


def _markdown_table(
    dataframe: pd.DataFrame,
    *,
    max_rows: int | None = None,
) -> str:
    """Convert a DataFrame into a Markdown table."""
    table = dataframe.copy()

    if max_rows is not None:
        table = table.head(max_rows)

    return table.to_markdown(index=False)


def create_evaluation_report(
    *,
    generation_results: pd.DataFrame,
    replicate_results: pd.DataFrame,
    output_directory: str | Path,
    title: str = "Phenotyping Strategy Evaluation",
) -> dict[str, Path]:
    """
    Run statistics, generate figures, and write a Markdown report.
    """
    output_directory = Path(
        output_directory
    ).expanduser().resolve()

    processed_directory = output_directory / "processed"
    figures_directory = output_directory / "figures"

    processed_directory.mkdir(
        parents=True,
        exist_ok=True,
    )
    figures_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    statistical_results = run_statistical_analysis(
        replicate_results,
        metrics=DEFAULT_METRICS,
    )

    statistics_paths = save_statistical_results(
        statistical_results,
        processed_directory,
    )

    figure_paths = create_all_plots(
        generation_results,
        replicate_results,
        figures_directory,
    )

    strategy_summary = (
        replicate_results.groupby("strategy")
        .agg(
            replicates=("replicate", "count"),
            mean_total_gain=(
                "total_realized_genetic_gain",
                "mean",
            ),
            median_total_gain=(
                "total_realized_genetic_gain",
                "median",
            ),
            mean_prediction_accuracy=(
                "mean_prediction_accuracy",
                "mean",
            ),
            mean_final_variance=(
                "final_genetic_variance",
                "mean",
            ),
            mean_variance_retention=(
                "variance_retention",
                "mean",
            ),
            mean_runtime_seconds=(
                "total_cycle_seconds",
                "mean",
            ),
        )
        .reset_index()
        .sort_values(
            "mean_total_gain",
            ascending=False,
        )
    )

    best_gain_row = strategy_summary.iloc[0]
    best_gain_strategy = str(
        best_gain_row["strategy"]
    )

    friedman = statistical_results[
        "friedman_tests"
    ]

    report_lines = [
        f"# {title}",
        "",
        "## Evaluation design",
        "",
        (
            f"The comparison contains "
            f"{replicate_results['replicate'].nunique()} matched "
            f"replicates, "
            f"{generation_results['generation'].nunique()} breeding "
            f"generations per replicate, and "
            f"{replicate_results['strategy'].nunique()} strategies."
        ),
        "",
        "The strategies were evaluated using the same initial population, "
        "phenotyping budget, parent-selection rule, crossing design, and "
        "replicate seeds.",
        "",
        "## Strategy summary",
        "",
        _markdown_table(
            strategy_summary.round(4)
        ),
        "",
        "## Main descriptive result",
        "",
        (
            f"The highest mean total realized genetic gain was observed "
            f"for **{best_gain_strategy}**, with a mean of "
            f"{_format_number(best_gain_row['mean_total_gain'])}."
        ),
        "",
        "This descriptive ranking should be interpreted together with the "
        "paired statistical tests and the retained genetic variance.",
        "",
        "## Omnibus repeated-measures tests",
        "",
    ]

    if friedman.empty:
        report_lines.append(
            "No Friedman tests were available."
        )
    else:
        report_lines.append(
            _markdown_table(
                friedman.round(5)
            )
        )

    report_lines.extend(
        [
            "",
            "## Figures",
            "",
        ]
    )

    for figure_name, figure_path in figure_paths.items():
        relative_path = figure_path.relative_to(
            output_directory
        )
        readable_name = figure_name.replace(
            "_",
            " ",
        ).title()

        report_lines.extend(
            [
                f"### {readable_name}",
                "",
                f"![{readable_name}]({relative_path.as_posix()})",
                "",
            ]
        )

    report_lines.extend(
        [
            "## Output tables",
            "",
            "- `raw/generation_results.csv`: one row per strategy, "
            "replicate, and generation.",
            "- `raw/replicate_results.csv`: one row per strategy and "
            "replicate.",
            "- `processed/descriptive_statistics.csv`: summary statistics "
            "for each outcome.",
            "- `processed/friedman_tests.csv`: overall repeated-measures "
            "tests.",
            "- `processed/pairwise_tests.csv`: paired Wilcoxon tests with "
            "Holm correction and effect sizes.",
            "",
            "## Interpretation cautions",
            "",
            "The development comparison with only a few replicates is a "
            "pipeline check, not the final scientific analysis. Final "
            "conclusions should use a larger number of matched replicates. "
            "A strategy should not be judged on genetic gain alone because "
            "rapid gain may be accompanied by faster depletion of genetic "
            "variance.",
            "",
        ]
    )

    report_path = output_directory / "evaluation_report.md"
    report_path.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    paths: dict[str, Path] = {
        "report": report_path,
        **{
            f"statistics_{name}": path
            for name, path in statistics_paths.items()
        },
        **{
            f"figure_{name}": path
            for name, path in figure_paths.items()
        },
    }

    return paths


def build_report_from_csv(
    evaluation_directory: str | Path,
) -> dict[str, Path]:
    """
    Build the complete report from previously saved raw CSV files.
    """
    evaluation_directory = Path(
        evaluation_directory
    ).expanduser().resolve()

    generation_path = (
        evaluation_directory
        / "raw"
        / "generation_results.csv"
    )
    replicate_path = (
        evaluation_directory
        / "raw"
        / "replicate_results.csv"
    )

    if not generation_path.is_file():
        raise FileNotFoundError(
            f"Missing generation results: {generation_path}"
        )

    if not replicate_path.is_file():
        raise FileNotFoundError(
            f"Missing replicate results: {replicate_path}"
        )

    generation_results = pd.read_csv(
        generation_path
    )
    replicate_results = pd.read_csv(
        replicate_path
    )

    return create_evaluation_report(
        generation_results=generation_results,
        replicate_results=replicate_results,
        output_directory=evaluation_directory,
    )


def main() -> None:
    """Build a report from results/evaluation/raw."""
    project_root = Path(__file__).resolve().parents[2]
    evaluation_directory = (
        project_root / "results" / "evaluation"
    )

    paths = build_report_from_csv(
        evaluation_directory
    )

    print("\nEvaluation report created.")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
