from pathlib import Path
from src.evaluation.compare_with_rl import compare_all_five_strategies, save_five_strategy_results
from src.evaluation.report import create_evaluation_report


def main():
    project_root = Path(__file__).resolve().parents[2]
    agent_path = (
        project_root
        / "results"
        / "rl"
        / "q_learning_development"
        / "q_learning_agent.pkl"
    )
    if not agent_path.is_file():
        raise FileNotFoundError(f"Trained agent not found: {agent_path}")

    output_directory = project_root / "results" / "evaluation_with_rl"

    result = compare_all_five_strategies(
        project_root=project_root,
        agent_path=agent_path,
        number_of_replicates=20,
        number_of_generations=20,
        number_to_phenotype=200,
        base_seed=20001,
    )

    raw_paths = save_five_strategy_results(result, output_directory)
    report_paths = create_evaluation_report(
        generation_results=result.generation_results,
        replicate_results=result.replicate_results,
        output_directory=output_directory,
        title="Phenotyping Strategy Evaluation Including RL",
    )

    print("\nFive-strategy evaluation completed.")
    for name, path in {**raw_paths, **report_paths}.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
