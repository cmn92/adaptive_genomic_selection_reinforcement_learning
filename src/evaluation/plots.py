"""
plots.py

Publication-style matplotlib figures for phenotyping-strategy evaluation.

The module intentionally uses matplotlib only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


STRATEGY_LABELS = {
    "random_sampling": "Random",
    "fixed_sampling": "Fixed positional",
    "diversity_sampling": "Diversity",
    "active_learning_pev": "Active learning (PEV)",
}


def _strategy_label(strategy: str) -> str:
    """Return a readable display label."""
    return STRATEGY_LABELS.get(
        strategy,
        strategy.replace("_", " ").title(),
    )


def _validate_generation_results(
    generation_results: pd.DataFrame,
) -> None:
    required = {"strategy", "replicate", "generation"}
    missing = required.difference(generation_results.columns)
    if missing:
        raise KeyError(
            "Generation results are missing required columns: "
            + ", ".join(sorted(missing))
        )


def _validate_replicate_results(
    replicate_results: pd.DataFrame,
) -> None:
    required = {"strategy", "replicate"}
    missing = required.difference(replicate_results.columns)
    if missing:
        raise KeyError(
            "Replicate results are missing required columns: "
            + ", ".join(sorted(missing))
        )


def plot_generation_trajectory(
    generation_results: pd.DataFrame,
    *,
    metric: str,
    ylabel: str,
    title: str,
    output_path: str | Path,
    confidence_level: float = 0.95,
) -> Path:
    """
    Plot strategy means across generations with normal-approximation CIs.
    """
    _validate_generation_results(generation_results)

    if metric not in generation_results.columns:
        raise KeyError(f"Metric '{metric}' is unavailable.")

    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = (
        generation_results.groupby(
            ["strategy", "generation"]
        )[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    z_value = 1.96 if confidence_level == 0.95 else 1.96

    summary["standard_error"] = (
        summary["std"] / np.sqrt(summary["count"])
    )
    summary["ci"] = z_value * summary["standard_error"]

    fig, ax = plt.subplots(figsize=(9, 6))

    for strategy, group in summary.groupby(
        "strategy",
        sort=True,
    ):
        group = group.sort_values("generation")

        x = group["generation"].to_numpy(dtype=float)
        mean = group["mean"].to_numpy(dtype=float)
        ci = group["ci"].fillna(0).to_numpy(dtype=float)

        ax.plot(
            x,
            mean,
            marker="o",
            label=_strategy_label(strategy),
        )
        ax.fill_between(
            x,
            mean - ci,
            mean + ci,
            alpha=0.15,
        )

    ax.set_title(title)
    ax.set_xlabel("Generation")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    return output_path


def plot_strategy_boxplot(
    replicate_results: pd.DataFrame,
    *,
    metric: str,
    ylabel: str,
    title: str,
    output_path: str | Path,
) -> Path:
    """Create a replicate-level boxplot for one metric."""
    _validate_replicate_results(replicate_results)

    if metric not in replicate_results.columns:
        raise KeyError(f"Metric '{metric}' is unavailable.")

    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    strategies = sorted(
        replicate_results["strategy"].unique()
    )

    data = [
        replicate_results.loc[
            replicate_results["strategy"] == strategy,
            metric,
        ].to_numpy(dtype=float)
        for strategy in strategies
    ]

    labels = [
        _strategy_label(strategy)
        for strategy in strategies
    ]

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.boxplot(
        data,
        labels=labels,
        showmeans=True,
    )

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=20)
    ax.grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    return output_path


def plot_gain_variance_tradeoff(
    replicate_results: pd.DataFrame,
    *,
    output_path: str | Path,
) -> Path:
    """Plot total gain against retained genetic variance."""
    _validate_replicate_results(replicate_results)

    required = {
        "total_realized_genetic_gain",
        "variance_retention",
    }
    missing = required.difference(replicate_results.columns)
    if missing:
        raise KeyError(
            "Trade-off plot requires: "
            + ", ".join(sorted(missing))
        )

    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))

    for strategy, group in replicate_results.groupby(
        "strategy",
        sort=True,
    ):
        ax.scatter(
            group["variance_retention"],
            group["total_realized_genetic_gain"],
            label=_strategy_label(strategy),
            alpha=0.75,
        )

    ax.set_title(
        "Genetic gain versus retained genetic variance"
    )
    ax.set_xlabel("Proportion of initial variance retained")
    ax.set_ylabel("Total realized genetic gain")
    ax.legend()
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    return output_path


def create_all_plots(
    generation_results: pd.DataFrame,
    replicate_results: pd.DataFrame,
    output_directory: str | Path,
) -> dict[str, Path]:
    """Generate the core evaluation figure set."""
    output_directory = Path(
        output_directory
    ).expanduser().resolve()
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths: dict[str, Path] = {}

    trajectory_specs = [
        (
            "next_generation_mean_gv",
            "Mean genetic value",
            "Mean genetic value across generations",
            "mean_genetic_value.png",
        ),
        (
            "prediction_accuracy",
            "Prediction accuracy",
            "Genomic prediction accuracy across generations",
            "prediction_accuracy.png",
        ),
        (
            "next_generation_variance_gv",
            "Genetic variance",
            "Genetic variance across generations",
            "genetic_variance.png",
        ),
    ]

    for metric, ylabel, title, filename in trajectory_specs:
        if metric in generation_results.columns:
            paths[filename.removesuffix(".png")] = (
                plot_generation_trajectory(
                    generation_results,
                    metric=metric,
                    ylabel=ylabel,
                    title=title,
                    output_path=output_directory / filename,
                )
            )

    boxplot_specs = [
        (
            "total_realized_genetic_gain",
            "Total realized genetic gain",
            "Total genetic gain by strategy",
            "total_genetic_gain_boxplot.png",
        ),
        (
            "variance_retention",
            "Variance retention",
            "Final variance retention by strategy",
            "variance_retention_boxplot.png",
        ),
        (
            "total_cycle_seconds",
            "Total cycle time (seconds)",
            "Computational time by strategy",
            "runtime_boxplot.png",
        ),
    ]

    for metric, ylabel, title, filename in boxplot_specs:
        if metric in replicate_results.columns:
            paths[filename.removesuffix(".png")] = (
                plot_strategy_boxplot(
                    replicate_results,
                    metric=metric,
                    ylabel=ylabel,
                    title=title,
                    output_path=output_directory / filename,
                )
            )

    if {
        "total_realized_genetic_gain",
        "variance_retention",
    }.issubset(replicate_results.columns):
        paths["gain_variance_tradeoff"] = (
            plot_gain_variance_tradeoff(
                replicate_results,
                output_path=(
                    output_directory
                    / "gain_variance_tradeoff.png"
                ),
            )
        )

    return paths
