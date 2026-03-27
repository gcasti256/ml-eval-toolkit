"""Model comparison example — evaluate two configs side-by-side."""

from ml_eval.config import EvalConfig, MetricConfig
from ml_eval.datasets.loader import load_dataset
from ml_eval.db import get_connection, init_db
from ml_eval.evaluation.comparison import compare_configs


def main() -> None:
    conn = get_connection(":memory:")
    init_db(conn)

    dataset = load_dataset("examples/sample_data.json")
    print(f"Loaded {len(dataset)} samples\n")

    # Define two evaluation configs with different metric parameters
    config_a = EvalConfig(
        dataset_path="examples/sample_data.json",
        metrics=[
            MetricConfig(name="bleu", params={"max_n": 4}),
            MetricConfig(name="rouge", params={"variant": "rouge-l"}),
        ],
        name="bleu4_rougeL",
    )

    config_b = EvalConfig(
        dataset_path="examples/sample_data.json",
        metrics=[
            MetricConfig(name="bleu", params={"max_n": 2}),
            MetricConfig(name="rouge", params={"variant": "rouge-1"}),
        ],
        name="bleu2_rouge1",
    )

    # Run comparison
    result = compare_configs(conn, dataset, [config_a, config_b])

    # Print comparison table
    print("=" * 60)
    print("Model Comparison Results")
    print("=" * 60)

    table = result.summary_table()
    if table:
        headers = list(table[0].keys())
        print("  ".join(f"{h:>15}" for h in headers))
        print("-" * (17 * len(headers)))
        for row in table:
            print("  ".join(f"{str(v):>15}" for v in row.values()))

    # Best config per metric
    print("\nBest config per metric:")
    all_metrics: set[str] = set()
    for r in result.results:
        all_metrics.update(r.aggregated.keys())
    for metric in sorted(all_metrics):
        best = result.best_config(metric)
        print(f"  {metric}: {best}")


if __name__ == "__main__":
    main()
