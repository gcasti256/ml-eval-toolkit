"""Basic evaluation example — load a dataset and run BLEU + ROUGE metrics."""

from ml_eval.config import EvalConfig, MetricConfig
from ml_eval.datasets.loader import load_dataset
from ml_eval.db import get_connection, init_db
from ml_eval.evaluation.runner import EvalRunner


def main() -> None:
    # Set up database
    conn = get_connection(":memory:")
    init_db(conn)

    # Load the sample dataset
    dataset = load_dataset("examples/sample_data.json")
    print(f"Loaded {len(dataset)} samples\n")

    # Configure evaluation with BLEU and ROUGE
    config = EvalConfig(
        dataset_path="examples/sample_data.json",
        metrics=[
            MetricConfig(name="bleu"),
            MetricConfig(name="rouge"),
        ],
        name="basic_eval",
    )

    # Run evaluation
    runner = EvalRunner(conn, config)
    result = runner.run(dataset)

    # Print results
    print("=" * 50)
    print(f"Evaluation: {result.name}")
    print(f"Run ID:     {result.run_id}")
    print("=" * 50)

    for metric, scores in result.aggregated.items():
        print(f"\n{metric.upper()}")
        print(f"  Average: {scores['avg']:.4f}")
        print(f"  Min:     {scores['min']:.4f}")
        print(f"  Max:     {scores['max']:.4f}")
        print(f"  Samples: {int(scores['count'])}")

    # Per-sample breakdown
    print("\n" + "-" * 50)
    print("Per-sample scores:")
    for metric, results_list in result.metric_results.items():
        print(f"\n  {metric}:")
        for i, mr in enumerate(results_list):
            print(f"    Sample {i}: {mr.score:.4f}")


if __name__ == "__main__":
    main()
