from pathlib import Path
from src.evaluation.run_rl_strategy import RLStrategyEvaluationConfig, evaluate_frozen_rl_strategy
from src.environment.state import observation_size
from src.rl.linear_q import LinearQAgent, LinearQConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def main():
    agent = LinearQAgent(
        number_of_actions=5,
        observation_size=observation_size(),
        config=LinearQConfig(seed=12345),
    )
    result = evaluate_frozen_rl_strategy(
        project_root=PROJECT_ROOT,
        agent=agent,
        discretizer=None,
        config=RLStrategyEvaluationConfig(
            number_of_replicates=1,
            number_of_generations=1,
            base_seed=20001,
            maximum_steps_per_episode=50,
        ),
    )
    assert len(result.generation_results) == 1
    assert len(result.replicate_results) == 1
    print(result.generation_results.to_string(index=False))
    print("\nFrozen RL evaluation check passed.")

if __name__ == "__main__":
    main()
