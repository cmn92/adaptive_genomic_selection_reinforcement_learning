"""
metrics.py

Metric calculations for comparing adaptive phenotyping strategies.

The functions in this module operate on generation-level results and return
one replicate-level summary row. They do not run simulations, plot figures,
or perform hypothesis tests.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


REQUIRED_GENERATION_COLUMNS = {
    "strategy",
    "generation",
    "number_phenotyped",
    "prediction_accuracy",
    "population_mean_gv_before",
    "population_variance_gv_before",
    "next_generation_mean_gv",
    "next_generation_variance_gv",
    "realized_genetic_gain",
}


def validate_generation_results(
    generation_results: pd.DataFrame,
) -> pd.DataFrame:
    """Validate and return a sorted copy of one strategy-replicate trajectory."""
    if not isinstance(generation_results, pd.DataFrame):
        raise TypeError("'generation_results' must be a pandas DataFrame.")

    if generation_results.empty:
        raise ValueError("'generation_results' cannot be empty.")

    missing = REQUIRED_GENERATION_COLUMNS.difference(
        generation_results.columns
    )
    if missing:
        raise KeyError(
            "Generation results are missing required columns: "
            + ", ".join(sorted(missing))
        )

    result = generation_results.copy()

    strategy_count = result["strategy"].nunique(dropna=False)
    if strategy_count != 1:
        raise ValueError(
            "Metric calculation expects exactly one strategy per input."
        )

    if "replicate" in result.columns:
        replicate_count = result["replicate"].nunique(dropna=False)
        if replicate_count != 1:
            raise ValueError(
                "Metric calculation expects exactly one replicate per input."
            )

    result = result.sort_values("generation").reset_index(drop=True)

    expected_generations = np.arange(
        int(result["generation"].iloc[0]),
        int(result["generation"].iloc[0]) + len(result),
    )
    observed_generations = result["generation"].to_numpy(dtype=int)

    if not np.array_equal(observed_generations, expected_generations):
        raise ValueError(
            "Generation numbers must be consecutive within a replicate."
        )

    numeric_columns = [
        "number_phenotyped",
        "prediction_accuracy",
        "population_mean_gv_before",
        "population_variance_gv_before",
        "next_generation_mean_gv",
        "next_generation_variance_gv",
        "realized_genetic_gain",
    ]

    for column in numeric_columns:
        values = pd.to_numeric(result[column], errors="coerce")
        if values.isna().any():
            raise ValueError(
                f"Column '{column}' contains missing or nonnumeric values."
            )
        result[column] = values

    return result


def area_under_gain_curve(
    generation_results: pd.DataFrame,
) -> float:
    """
    Calculate area under the genetic-gain trajectory.

    The curve is defined relative to the initial population mean genetic value.
    The initial point at generation zero therefore has gain equal to zero.
    """
    results = validate_generation_results(generation_results)

    initial_mean = float(
        results.loc[0, "population_mean_gv_before"]
    )

    cumulative_gain = (
        results["next_generation_mean_gv"].to_numpy(dtype=float)
        - initial_mean
    )

    x = np.arange(
        0,
        len(cumulative_gain) + 1,
        dtype=float,
    )
    y = np.concatenate(
        [np.array([0.0]), cumulative_gain]
    )

    return float(np.trapz(y, x))


def rate_of_variance_loss(
    generation_results: pd.DataFrame,
) -> float:
    """
    Estimate the per-generation slope of genetic variance.

    A more negative value means genetic variance declined more quickly.
    """
    results = validate_generation_results(generation_results)

    x = np.concatenate(
        [
            np.array([0.0]),
            results["generation"].to_numpy(dtype=float),
        ]
    )

    y = np.concatenate(
        [
            np.array(
                [
                    float(
                        results.loc[
                            0,
                            "population_variance_gv_before",
                        ]
                    )
                ]
            ),
            results[
                "next_generation_variance_gv"
            ].to_numpy(dtype=float),
        ]
    )

    if np.allclose(x, x[0]):
        return np.nan

    slope = np.polyfit(x, y, deg=1)[0]
    return float(slope)


def summarize_strategy_replicate(
    generation_results: pd.DataFrame,
    *,
    replicate: int | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Convert one generation trajectory into one replicate-level summary row.
    """
    results = validate_generation_results(generation_results)

    strategy = str(results.loc[0, "strategy"])
    first = results.iloc[0]
    last = results.iloc[-1]

    if replicate is None and "replicate" in results.columns:
        replicate = int(results.loc[0, "replicate"])

    if seed is None:
        if "run_seed" in results.columns:
            seed = int(results.loc[0, "run_seed"])
        elif "seed" in results.columns:
            seed = int(results.loc[0, "seed"])

    initial_mean = float(first["population_mean_gv_before"])
    final_mean = float(last["next_generation_mean_gv"])
    total_gain = final_mean - initial_mean

    initial_variance = float(
        first["population_variance_gv_before"]
    )
    final_variance = float(
        last["next_generation_variance_gv"]
    )

    variance_retention = (
        final_variance / initial_variance
        if initial_variance > 0
        else np.nan
    )

    total_phenotyped = int(
        results["number_phenotyped"].sum()
    )

    gain_per_100 = (
        total_gain / total_phenotyped * 100
        if total_phenotyped > 0
        else np.nan
    )

    prediction_accuracy = results[
        "prediction_accuracy"
    ].to_numpy(dtype=float)

    summary: dict[str, Any] = {
        "strategy": strategy,
        "replicate": replicate,
        "seed": seed,
        "number_of_generations": int(len(results)),
        "total_number_phenotyped": total_phenotyped,
        "initial_mean_genetic_value": initial_mean,
        "final_mean_genetic_value": final_mean,
        "total_realized_genetic_gain": total_gain,
        "mean_gain_per_generation": float(
            results["realized_genetic_gain"].mean()
        ),
        "gain_standard_deviation": float(
            results["realized_genetic_gain"].std(ddof=1)
        )
        if len(results) > 1
        else 0.0,
        "gain_per_100_phenotypes": gain_per_100,
        "area_under_genetic_gain_curve": (
            area_under_gain_curve(results)
        ),
        "mean_prediction_accuracy": float(
            np.mean(prediction_accuracy)
        ),
        "final_prediction_accuracy": float(
            prediction_accuracy[-1]
        ),
        "minimum_prediction_accuracy": float(
            np.min(prediction_accuracy)
        ),
        "maximum_prediction_accuracy": float(
            np.max(prediction_accuracy)
        ),
        "prediction_accuracy_variability": float(
            np.std(prediction_accuracy, ddof=1)
        )
        if len(prediction_accuracy) > 1
        else 0.0,
        "initial_genetic_variance": initial_variance,
        "final_genetic_variance": final_variance,
        "variance_retention": variance_retention,
        "rate_of_variance_loss": rate_of_variance_loss(
            results
        ),
    }

    optional_sum_columns = {
        "phenotyping_cost_units": (
            "cumulative_phenotyping_cost"
        ),
    }
    for source_column, output_column in optional_sum_columns.items():
        if source_column in results.columns:
            summary[output_column] = float(
                results[source_column].sum()
            )

    optional_mean_columns = [
        "selection_seconds",
        "cycle_seconds",
        "initial_model_accuracy",
        "final_model_accuracy",
    ]
    for column in optional_mean_columns:
        if column in results.columns:
            summary[f"mean_{column}"] = float(
                pd.to_numeric(
                    results[column],
                    errors="coerce",
                ).mean()
            )

    if "cycle_seconds" in results.columns:
        summary["total_cycle_seconds"] = float(
            pd.to_numeric(
                results["cycle_seconds"],
                errors="coerce",
            ).sum()
        )

    return pd.DataFrame([summary])


def summarize_all_replicates(
    generation_results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize all strategy-replicate groups in a combined generation table.
    """
    if not isinstance(generation_results, pd.DataFrame):
        raise TypeError("'generation_results' must be a pandas DataFrame.")

    required = {"strategy", "replicate"}
    missing = required.difference(generation_results.columns)
    if missing:
        raise KeyError(
            "Combined generation results are missing: "
            + ", ".join(sorted(missing))
        )

    rows: list[pd.DataFrame] = []

    for (strategy, replicate), group in generation_results.groupby(
        ["strategy", "replicate"],
        sort=True,
    ):
        seed = None
        if "run_seed" in group.columns:
            seed = int(group["run_seed"].iloc[0])
        elif "seed" in group.columns:
            seed = int(group["seed"].iloc[0])

        rows.append(
            summarize_strategy_replicate(
                group,
                replicate=int(replicate),
                seed=seed,
            )
        )

    return pd.concat(rows, ignore_index=True)
