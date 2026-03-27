"""Command-line interface for ML Eval Toolkit."""

from __future__ import annotations

import json
import sys

import click
from rich.console import Console
from rich.table import Table

from ml_eval import __version__
from ml_eval.config import EvalConfig, MetricConfig, load_config
from ml_eval.datasets.loader import load_dataset
from ml_eval.db import (
    get_connection,
    init_db,
    list_runs,
    save_baseline,
)
from ml_eval.evaluation.comparison import compare_configs
from ml_eval.evaluation.regression import check_regression
from ml_eval.evaluation.runner import EvalRunner
from ml_eval.reporting.exporter import export_csv, export_json

console = Console()


@click.group()
@click.version_option(version=__version__)
def cli() -> None:
    """ML Eval Toolkit — Evaluate and benchmark ML model outputs."""


@cli.command()
@click.option("--dataset", "-d", required=True, help="Path to dataset file (CSV, JSON, JSONL)")
@click.option("--metrics", "-m", required=True, help="Comma-separated metric names (bleu,rouge,semantic,rubric,llm_judge)")
@click.option("--name", "-n", default="cli_eval", help="Name for this evaluation run")
@click.option("--output", "-o", default=None, help="Output file path (JSON or CSV based on extension)")
@click.option("--save-baseline", "baseline_name", default=None, help="Save results as a named baseline")
def run(dataset: str, metrics: str, name: str, output: str | None, baseline_name: str | None) -> None:
    """Run evaluation metrics against a dataset."""
    conn = get_connection()
    init_db(conn)

    metric_names = [m.strip() for m in metrics.split(",") if m.strip()]
    config = EvalConfig(
        dataset_path=dataset,
        metrics=[MetricConfig(name=m) for m in metric_names],
        name=name,
    )

    try:
        ds = load_dataset(dataset)
        console.print(f"[bold]Loaded {len(ds)} samples from {dataset}[/bold]")
    except (FileNotFoundError, ValueError) as e:
        console.print(f"[red]Error loading dataset: {e}[/red]")
        sys.exit(1)

    try:
        runner = EvalRunner(conn, config)
        result = runner.run(ds)
    except ValueError as e:
        console.print(f"[red]Error during evaluation: {e}[/red]")
        sys.exit(1)

    # Display results
    table = Table(title=f"Evaluation Results: {name}")
    table.add_column("Metric", style="cyan")
    table.add_column("Avg Score", style="green")
    table.add_column("Min", style="yellow")
    table.add_column("Max", style="yellow")
    table.add_column("Samples", style="dim")

    for metric, agg in result.aggregated.items():
        table.add_row(
            metric,
            f"{agg['avg']:.4f}",
            f"{agg['min']:.4f}",
            f"{agg['max']:.4f}",
            str(int(agg["count"])),
        )

    console.print(table)
    console.print(f"\n[dim]Run ID: {result.run_id}[/dim]")

    if output:
        if output.endswith(".csv"):
            export_csv(conn, result.run_id, output)
        else:
            export_json(conn, result.run_id, output)
        console.print(f"[green]Results exported to {output}[/green]")

    if baseline_name:
        save_baseline(conn, result.run_id, baseline_name)
        console.print(f"[green]Baseline saved as '{baseline_name}'[/green]")


@cli.command()
@click.option("--dataset", "-d", required=True, help="Path to dataset file")
@click.option("--configs", "-c", required=True, multiple=True, help="Paths to config YAML files")
def compare(dataset: str, configs: tuple[str, ...]) -> None:
    """Run side-by-side comparison of multiple evaluation configs."""
    conn = get_connection()
    init_db(conn)

    try:
        ds = load_dataset(dataset)
    except (FileNotFoundError, ValueError) as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)

    eval_configs = []
    for path in configs:
        try:
            cfg = load_config(path)
            cfg.dataset_path = dataset
            eval_configs.append(cfg)
        except Exception as e:
            console.print(f"[red]Error loading config {path}: {e}[/red]")
            sys.exit(1)

    result = compare_configs(conn, ds, eval_configs)
    comparison_table = result.summary_table()

    if comparison_table:
        table = Table(title="Model Comparison")
        for key in comparison_table[0]:
            table.add_column(key.replace("_", " ").title(), style="cyan")
        for row in comparison_table:
            table.add_row(*[str(v) for v in row.values()])
        console.print(table)


@cli.command()
@click.option("--dataset", "-d", required=True, help="Path to dataset file")
@click.option("--metrics", "-m", required=True, help="Comma-separated metric names")
@click.option("--baseline", "-b", required=True, help="Baseline name to compare against")
@click.option("--threshold", "-t", default=0.05, type=float, help="Regression threshold (default 0.05)")
@click.option("--name", "-n", default="regression_check", help="Name for this run")
def regression(dataset: str, metrics: str, baseline: str, threshold: float, name: str) -> None:
    """Run regression testing against a stored baseline."""
    conn = get_connection()
    init_db(conn)

    metric_names = [m.strip() for m in metrics.split(",") if m.strip()]
    config = EvalConfig(
        dataset_path=dataset,
        metrics=[MetricConfig(name=m) for m in metric_names],
        name=name,
    )

    try:
        ds = load_dataset(dataset)
        runner = EvalRunner(conn, config)
        result = runner.run(ds)
    except (FileNotFoundError, ValueError) as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)

    try:
        reg_result = check_regression(conn, result.run_id, baseline, threshold)
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)

    if reg_result.has_regression:
        console.print("[bold red]REGRESSIONS DETECTED[/bold red]")
        for r in reg_result.regressions:
            console.print(
                f"  {r['metric']}: {r['baseline_avg']:.4f} → {r['current_avg']:.4f} "
                f"({r['delta']:+.4f}, {r['percent_change']:+.1f}%)"
            )
        sys.exit(1)
    else:
        console.print("[bold green]No regressions detected[/bold green]")
        for imp in reg_result.improvements:
            console.print(
                f"  [green]{imp['metric']}: {imp['baseline_avg']:.4f} → {imp['current_avg']:.4f} "
                f"({imp['delta']:+.4f}, {imp['percent_change']:+.1f}%)[/green]"
            )


@cli.command()
@click.option("--format", "fmt", type=click.Choice(["table", "json"]), default="table", help="Output format")
@click.option("--limit", "-l", default=20, help="Number of results to show")
def results(fmt: str, limit: int) -> None:
    """List stored evaluation results."""
    conn = get_connection()
    init_db(conn)
    runs = list_runs(conn, limit=limit)

    if not runs:
        console.print("[dim]No evaluation runs found.[/dim]")
        return

    if fmt == "json":
        console.print(json.dumps(runs, indent=2, default=str))
        return

    table = Table(title="Evaluation Runs")
    table.add_column("Run ID", style="dim", max_width=36)
    table.add_column("Name", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Created", style="dim")

    for run in runs:
        status_style = "green" if run["status"] == "completed" else "red"
        table.add_row(
            run["id"],
            run["name"],
            f"[{status_style}]{run['status']}[/{status_style}]",
            run["created_at"][:19],
        )

    console.print(table)
