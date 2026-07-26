"""Unit tests for evaluation metric calculations."""

import numpy as np
import pandas as pd

from src.evaluation.metrics import summarize_strategy_replicate


def main() -> None:
    generation_results = pd.DataFrame(
        {
            "strategy": ["degenerate_case", "degenerate_case"],
            "replicate": [1, 1],
            "generation": [1, 2],
            "number_phenotyped": [200, 200],
            "prediction_accuracy": [np.nan, 0.0],
            "population_mean_gv_before": [1.0, 1.1],
            "population_variance_gv_before": [0.8, 0.0],
            "next_generation_mean_gv": [1.1, 1.1],
            "next_generation_variance_gv": [0.0, 0.0],
            "realized_genetic_gain": [0.1, 0.0],
        }
    )

    summary = summarize_strategy_replicate(
        generation_results
    )

    assert len(summary) == 1
    assert summary.loc[0, "mean_prediction_accuracy"] == 0.0
    assert summary.loc[0, "final_prediction_accuracy"] == 0.0
    assert np.isfinite(
        summary.loc[0, "total_realized_genetic_gain"]
    )

    print("All metric checks passed.")


if __name__ == "__main__":
    main()
