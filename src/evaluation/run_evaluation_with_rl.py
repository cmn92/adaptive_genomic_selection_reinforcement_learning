import argparse
from pathlib import Path

from src.evaluation.compare_with_rl import compare_all_five_strategies, save_five_strategy_results
from src.evaluation.report import create_evaluation_report


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate fixed baselines with a frozen RL agent."
    )
    parser.add_argument("--heritability", type=float, default=0.05)
    parser.add_argument("--population-size", type=int, default=500)
    parser.add_argument("--replicates", type=int, default=20)
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--maximum-phenotypes", type=int, default=75)
    parser.add_argument("--number-of-parents", type=int, default=20)
    parser.add_argument("--number-of-crosses", type=int, default=50)
    parser.add_argument("--dh-per-f1", type=int, default=10)
    parser.add_argument("--phenotype-reps", type=int, default=2)
    parser.add_argument("--base-seed", type=int, default=30001)
    parser.add_argument("--agent-path", default=None)
    parser.add_argument("--output-directory", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    if args.agent_path is None:
        heritability_label = str(args.heritability).replace(".", "_")
        agent_path = (
            project_root
            / "results"
            / "rl"
            / f"linear_q_h2_{heritability_label}"
            / "q_learning_agent.pkl"
        )
    else:
        agent_path = Path(args.agent_path)
        if not agent_path.is_absolute():
            agent_path = project_root / agent_path

    if not agent_path.is_file():
        raise FileNotFoundError(f"Trained agent not found: {agent_path}")

    if args.output_directory is None:
        heritability_label = str(args.heritability).replace(".", "_")
        output_directory = (
            project_root
            / "results"
            / f"evaluation_with_rl_h2_{heritability_label}"
        )
    else:
        output_directory = Path(args.output_directory)
        if not output_directory.is_absolute():
            output_directory = project_root / output_directory

    result = compare_all_five_strategies(
        project_root=project_root,
        agent_path=agent_path,
        number_of_replicates=args.replicates,
        number_of_generations=args.generations,
        number_to_phenotype=args.maximum_phenotypes,
        base_seed=args.base_seed,
        trait_heritability=args.heritability,
        population_size=args.population_size,
        number_of_parents=args.number_of_parents,
        number_of_crosses=args.number_of_crosses,
        dh_per_f1=args.dh_per_f1,
        reps=args.phenotype_reps,
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
