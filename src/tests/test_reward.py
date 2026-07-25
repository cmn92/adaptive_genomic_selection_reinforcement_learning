"""Unit tests for reward calculations."""

import numpy as np

from src.environment.reward import (
    RewardConfig,
    batch_cost_reward,
    final_generation_reward,
    invalid_action_reward,
    model_quality_reward,
)


def main() -> None:
    config = RewardConfig(
        genetic_gain_weight=1.0,
        variance_retention_weight=0.5,
        phenotyping_cost_weight=0.2,
        reliability_improvement_weight=0.3,
        invalid_action_penalty=1.0,
        gain_scale=1.0,
    )

    batch = batch_cost_reward(
        batch_size=25,
        maximum_phenotypes=200,
        config=config,
    )

    assert np.isclose(batch.total, -0.025)
    assert batch.cost_component < 0

    strong = final_generation_reward(
        realized_genetic_gain=1.5,
        variance_retention=0.5,
        number_phenotyped=100,
        maximum_phenotypes=200,
        config=config,
    )

    weak = final_generation_reward(
        realized_genetic_gain=0.2,
        variance_retention=0.1,
        number_phenotyped=200,
        maximum_phenotypes=200,
        config=config,
    )

    assert strong.total > weak.total
    assert strong.genetic_gain_component > 0
    assert strong.variance_component > 0
    assert strong.cost_component < 0

    model_quality = model_quality_reward(
        previous_mean_reliability=0.2,
        current_mean_reliability=0.5,
        config=config,
    )

    assert model_quality.total > 0
    assert model_quality.model_quality_component > 0

    invalid = invalid_action_reward(config)
    assert invalid.total == -1.0
    assert invalid.invalid_action_component == -1.0

    print("All reward checks passed.")


if __name__ == "__main__":
    main()
