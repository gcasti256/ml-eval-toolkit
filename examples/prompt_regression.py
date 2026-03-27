"""Prompt regression testing — detect score degradations against a baseline."""

from ml_eval.config import EvalConfig, MetricConfig
from ml_eval.datasets.loader import load_dataset
from ml_eval.db import get_connection, init_db, save_baseline
from ml_eval.evaluation.regression import check_regression
from ml_eval.evaluation.runner import EvalRunner


def main() -> None:
    conn = get_connection(":memory:")
    init_db(conn)

    dataset = load_dataset("examples/sample_data.json")

    # Step 1: Run initial evaluation and save as baseline
    print("Step 1: Establishing baseline...")
    baseline_config = EvalConfig(
        dataset_path="examples/sample_data.json",
        metrics=[
            MetricConfig(name="bleu"),
            MetricConfig(name="rouge"),
        ],
        name="baseline_run",
    )

    runner = EvalRunner(conn, baseline_config)
    baseline_result = runner.run(dataset)
    save_baseline(conn, baseline_result.run_id, "v1_baseline")

    print(f"Baseline saved (run_id: {baseline_result.run_id})")
    for metric, scores in baseline_result.aggregated.items():
        print(f"  {metric}: {scores['avg']:.4f}")

    # Step 2: Run a new evaluation (simulating a prompt change)
    print("\nStep 2: Running current evaluation...")
    current_config = EvalConfig(
        dataset_path="examples/sample_data.json",
        metrics=[
            MetricConfig(name="bleu"),
            MetricConfig(name="rouge"),
        ],
        name="current_run",
    )

    current_runner = EvalRunner(conn, current_config)
    current_result = current_runner.run(dataset)

    for metric, scores in current_result.aggregated.items():
        print(f"  {metric}: {scores['avg']:.4f}")

    # Step 3: Check for regressions
    print("\nStep 3: Checking for regressions (threshold: 5%)...")
    reg_result = check_regression(
        conn,
        current_run_id=current_result.run_id,
        baseline_name="v1_baseline",
        threshold=0.05,
    )

    if reg_result.has_regression:
        print("\nREGRESSIONS DETECTED:")
        for r in reg_result.regressions:
            print(
                f"  {r['metric']}: {r['baseline_avg']:.4f} -> {r['current_avg']:.4f} "
                f"({r['delta']:+.4f}, {r['percent_change']:+.1f}%)"
            )
    else:
        print("\nNo regressions detected.")

    if reg_result.improvements:
        print("\nImprovements:")
        for imp in reg_result.improvements:
            print(
                f"  {imp['metric']}: {imp['baseline_avg']:.4f} -> {imp['current_avg']:.4f} "
                f"({imp['delta']:+.4f}, {imp['percent_change']:+.1f}%)"
            )


if __name__ == "__main__":
    main()
