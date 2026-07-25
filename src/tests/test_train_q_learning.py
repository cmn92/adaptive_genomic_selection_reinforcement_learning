"""
Small integration test for Q-learning training.

This test deliberately uses one episode and one breeding generation because
full RL training is computationally expensive.
"""

from pathlib import Path

from src.environment.breeding_env import (
    BreedingEnv,
    BreedingEnvConfig,
)
from src.environment.r_bridge import RBreedingBridge
from src.environment.reward import RewardConfig
from src.environment.state import observation_size
from src.rl.discretizer import ObservationDiscretizer
from src.rl.q_learning import (
    QLearningAgent,
    QLearningConfig,
)
from src.rl.train import (
    TrainingConfig,
    train_q_learning,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    bridge = RBreedingBridge(
        project_root=PROJECT_ROOT,
        seed=12345,
    )

    env = BreedingEnv(
        bridge=bridge,
        config=BreedingEnvConfig(
            maximum_generations=1,
            batch_size=25,
            minimum_training_size=50,
            maximum_phenotypes=50,
            seed=12345,
        ),
        reward_config=RewardConfig(),
    )

    discretizer = ObservationDiscretizer(
        bins_per_feature=3,
        observation_size=observation_size(),
    )

    agent = QLearningAgent(
        number_of_actions=env.action_space.n,
        config=QLearningConfig(
            learning_rate=0.2,
            discount_factor=0.95,
            epsilon_start=1.0,
            epsilon_end=0.1,
            epsilon_decay_episodes=10,
            seed=12345,
        ),
    )

    result = train_q_learning(
        env=env,
        agent=agent,
        discretizer=discretizer,
        config=TrainingConfig(
            number_of_episodes=1,
            maximum_steps_per_episode=10,
            seed=12345,
            checkpoint_every=1,
        ),
    )

    assert len(result.episode_history) == 1
    assert len(result.agent.q_table) > 0
    assert result.total_runtime_seconds > 0

    print("\nTraining history:")
    print(
        result.episode_history.to_string(
            index=False
        )
    )
    print("\nAll training checks passed.")


if __name__ == "__main__":
    main()
