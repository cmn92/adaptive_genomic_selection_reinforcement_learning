"""
statistics.py

Repeated-measures statistical comparisons for phenotyping strategies.

The primary workflow is:

1. read replicate_results.csv;
2. describe each strategy;
3. run a Friedman test across all strategies;
4. run paired Wilcoxon signed-rank tests;
5. apply Holm multiplicity correction;
6. report paired effect sizes and bootstrap confidence intervals.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from scipy.stats import (
        friedmanchisquare,
        wilcoxon,
    )
except ImportError as exc:
    raise ImportError(
        "scipy is required for statistical comparisons. "
        "Install it with: python -m pip install scipy"
    ) from exc


DEFAULT_METRICS = (
    "total_realized_genetic_gain",
    "final_mean_genetic_value",
    "mean_prediction_accuracy",
    "final_genetic_variance",
    "variance_retention",
    "total_cycle_seconds",
)


def validate_replicate_results(
    replicate_results: pd.DataFrame,
    metric: str,
) -> pd.DataFrame:
    """Validate a replicate-level comparison table."""
    if not isinstance(replicate_results, pd.DataFrame):
        raise TypeError(
            "'replicate_results' must be a pandas DataFrame."
        )

    required = {"strategy", "replicate", metric}
    missing = required.difference(replicate_results.columns)
    if missing:
        raise KeyError(
            "Replicate results are missing required columns: "
            + ", ".join(sorted(missing))
        )

    result = replicate_results[
        ["strategy", "replicate", metric]
    ].copy()

    result[metric] = pd.to_numeric(
        result[metric],
        errors="coerce",
    )

    if result[metric].isna().any():
        raise ValueError(
            f"Metric '{metric}' contains missing or nonnumeric values."
        )

    duplicate_mask = result.duplicated(
        ["strategy", "replicate"],
        keep=False,
    )
    if duplicate_mask.any():
        raise ValueError(
            "There must be one row per strategy and replicate."
        )

    return result


def descriptive_statistics(
    replicate_results: pd.DataFrame,
    metric: str,
) -> pd.DataFrame:
    """Calculate descriptive statistics by strategy."""
    data = validate_replicate_results(
        replicate_results,
        metric,
    )

    summary = (
        data.groupby("strategy")[metric]
        .agg(
            count="count",
            mean="mean",
            standard_deviation="std",
            median="median",
            minimum="min",
            maximum="max",
        )
        .reset_index()
    )

    quantiles = (
        data.groupby("strategy")[metric]
        .quantile([0.25, 0.75])
        .unstack()
        .rename(columns={0.25: "q1", 0.75: "q3"})
        .reset_index()
    )

    summary = summary.merge(
        quantiles,
        on="strategy",
        how="left",
    )
    summary["metric"] = metric

    return summary[
        [
            "metric",
            "strategy",
            "count",
            "mean",
            "standard_deviation",
            "median",
            "q1",
            "q3",
            "minimum",
            "maximum",
        ]
    ]


def _paired_matrix(
    replicate_results: pd.DataFrame,
    metric: str,
) -> pd.DataFrame:
    """Create a complete replicate-by-strategy matrix."""
    data = validate_replicate_results(
        replicate_results,
        metric,
    )

    matrix = data.pivot(
        index="replicate",
        columns="strategy",
        values=metric,
    ).sort_index()

    if matrix.isna().any().any():
        missing = matrix.isna().sum()
        raise ValueError(
            "The repeated-measures comparison is incomplete. "
            "Missing observations by strategy: "
            + ", ".join(
                f"{name}={count}"
                for name, count in missing.items()
                if count > 0
            )
        )

    if matrix.shape[1] < 2:
        raise ValueError(
            "At least two strategies are required."
        )

    return matrix


def friedman_test(
    replicate_results: pd.DataFrame,
    metric: str,
) -> pd.DataFrame:
    """Run the Friedman repeated-measures test."""
    matrix = _paired_matrix(
        replicate_results,
        metric,
    )

    if matrix.shape[1] < 3:
        raise ValueError(
            "The Friedman test requires at least three strategies."
        )

    arrays = [
        matrix[column].to_numpy(dtype=float)
        for column in matrix.columns
    ]

    statistic, p_value = friedmanchisquare(*arrays)

    n = matrix.shape[0]
    k = matrix.shape[1]

    kendalls_w = (
        statistic / (n * (k - 1))
        if n > 0 and k > 1
        else np.nan
    )

    return pd.DataFrame(
        {
            "metric": [metric],
            "test": ["Friedman"],
            "number_of_replicates": [n],
            "number_of_strategies": [k],
            "statistic": [float(statistic)],
            "p_value": [float(p_value)],
            "kendalls_w": [float(kendalls_w)],
        }
    )


def holm_adjust(
    p_values: Iterable[float],
) -> np.ndarray:
    """Apply the Holm step-down family-wise error correction."""
    values = np.asarray(list(p_values), dtype=float)

    if values.ndim != 1:
        raise ValueError("'p_values' must be one-dimensional.")

    if np.any((values < 0) | (values > 1)):
        raise ValueError("P-values must lie between zero and one.")

    m = len(values)
    order = np.argsort(values)
    ordered = values[order]

    adjusted_ordered = np.empty(m, dtype=float)
    running_max = 0.0

    for rank, p_value in enumerate(ordered):
        adjusted = (m - rank) * p_value
        running_max = max(running_max, adjusted)
        adjusted_ordered[rank] = min(running_max, 1.0)

    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adjusted_ordered

    return adjusted


def paired_rank_biserial(
    differences: np.ndarray,
) -> float:
    """
    Calculate matched-pairs rank-biserial correlation.

    Positive values favor the first strategy in the pair.
    """
    differences = np.asarray(
        differences,
        dtype=float,
    )
    differences = differences[
        ~np.isclose(differences, 0.0)
    ]

    if differences.size == 0:
        return 0.0

    absolute = np.abs(differences)
    ranks = pd.Series(absolute).rank(
        method="average"
    ).to_numpy(dtype=float)

    positive_rank_sum = ranks[
        differences > 0
    ].sum()
    negative_rank_sum = ranks[
        differences < 0
    ].sum()

    denominator = positive_rank_sum + negative_rank_sum

    if denominator == 0:
        return 0.0

    return float(
        (positive_rank_sum - negative_rank_sum)
        / denominator
    )


def bootstrap_mean_difference_ci(
    first: np.ndarray,
    second: np.ndarray,
    *,
    confidence_level: float = 0.95,
    n_bootstrap: int = 10000,
    seed: int = 12345,
) -> tuple[float, float]:
    """Bootstrap a percentile interval for the paired mean difference."""
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)

    if first.shape != second.shape:
        raise ValueError(
            "Paired arrays must have identical shapes."
        )

    if first.ndim != 1 or first.size < 2:
        raise ValueError(
            "At least two paired observations are required."
        )

    if not 0 < confidence_level < 1:
        raise ValueError(
            "'confidence_level' must lie strictly between 0 and 1."
        )

    if n_bootstrap < 100:
        raise ValueError(
            "'n_bootstrap' must be at least 100."
        )

    differences = first - second
    rng = np.random.default_rng(seed)

    sampled_indices = rng.integers(
        0,
        differences.size,
        size=(n_bootstrap, differences.size),
    )

    bootstrap_means = differences[
        sampled_indices
    ].mean(axis=1)

    alpha = 1 - confidence_level

    lower, upper = np.quantile(
        bootstrap_means,
        [alpha / 2, 1 - alpha / 2],
    )

    return float(lower), float(upper)


def pairwise_wilcoxon_tests(
    replicate_results: pd.DataFrame,
    metric: str,
    *,
    confidence_level: float = 0.95,
    n_bootstrap: int = 10000,
    bootstrap_seed: int = 12345,
) -> pd.DataFrame:
    """Run all paired Wilcoxon comparisons with Holm correction."""
    matrix = _paired_matrix(
        replicate_results,
        metric,
    )

    rows: list[dict[str, float | int | str]] = []

    for pair_index, (first_name, second_name) in enumerate(
        combinations(matrix.columns, 2)
    ):
        first = matrix[first_name].to_numpy(dtype=float)
        second = matrix[second_name].to_numpy(dtype=float)
        differences = first - second

        if np.allclose(differences, 0.0):
            statistic = 0.0
            p_value = 1.0
        else:
            test = wilcoxon(
                first,
                second,
                alternative="two-sided",
                zero_method="wilcox",
                correction=False,
                method="auto",
            )
            statistic = float(test.statistic)
            p_value = float(test.pvalue)

        ci_lower, ci_upper = bootstrap_mean_difference_ci(
            first,
            second,
            confidence_level=confidence_level,
            n_bootstrap=n_bootstrap,
            seed=bootstrap_seed + pair_index,
        )

        rows.append(
            {
                "metric": metric,
                "strategy_1": first_name,
                "strategy_2": second_name,
                "number_of_pairs": len(first),
                "mean_strategy_1": float(np.mean(first)),
                "mean_strategy_2": float(np.mean(second)),
                "mean_paired_difference": float(
                    np.mean(differences)
                ),
                "median_paired_difference": float(
                    np.median(differences)
                ),
                "bootstrap_ci_lower": ci_lower,
                "bootstrap_ci_upper": ci_upper,
                "wilcoxon_statistic": statistic,
                "p_value_unadjusted": p_value,
                "rank_biserial_correlation": (
                    paired_rank_biserial(differences)
                ),
            }
        )

    results = pd.DataFrame(rows)

    results["p_value_holm"] = holm_adjust(
        results["p_value_unadjusted"]
    )
    results["significant_holm_0_05"] = (
        results["p_value_holm"] < 0.05
    )

    return results


def run_statistical_analysis(
    replicate_results: pd.DataFrame,
    *,
    metrics: Iterable[str] = DEFAULT_METRICS,
    confidence_level: float = 0.95,
    n_bootstrap: int = 10000,
    bootstrap_seed: int = 12345,
) -> dict[str, pd.DataFrame]:
    """Run descriptive, omnibus, and pairwise analyses for each metric."""
    descriptive_tables: list[pd.DataFrame] = []
    friedman_tables: list[pd.DataFrame] = []
    pairwise_tables: list[pd.DataFrame] = []

    for metric in metrics:
        if metric not in replicate_results.columns:
            continue

        descriptive_tables.append(
            descriptive_statistics(
                replicate_results,
                metric,
            )
        )

        strategy_count = replicate_results[
            "strategy"
        ].nunique()

        if strategy_count >= 3:
            friedman_tables.append(
                friedman_test(
                    replicate_results,
                    metric,
                )
            )

        pairwise_tables.append(
            pairwise_wilcoxon_tests(
                replicate_results,
                metric,
                confidence_level=confidence_level,
                n_bootstrap=n_bootstrap,
                bootstrap_seed=bootstrap_seed,
            )
        )

    return {
        "descriptive_statistics": pd.concat(
            descriptive_tables,
            ignore_index=True,
        )
        if descriptive_tables
        else pd.DataFrame(),
        "friedman_tests": pd.concat(
            friedman_tables,
            ignore_index=True,
        )
        if friedman_tables
        else pd.DataFrame(),
        "pairwise_tests": pd.concat(
            pairwise_tables,
            ignore_index=True,
        )
        if pairwise_tables
        else pd.DataFrame(),
    }


def save_statistical_results(
    results: dict[str, pd.DataFrame],
    output_directory: str | Path,
) -> dict[str, Path]:
    """Save all statistical output tables."""
    output_directory = Path(
        output_directory
    ).expanduser().resolve()

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths: dict[str, Path] = {}

    for name, table in results.items():
        path = output_directory / f"{name}.csv"
        table.to_csv(path, index=False)
        paths[name] = path

    return paths
