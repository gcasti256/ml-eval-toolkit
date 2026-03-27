"""Tests for the CLI interface."""

import json
from pathlib import Path

from click.testing import CliRunner

from ml_eval.cli import cli


def _make_dataset(tmp_path: Path) -> Path:
    data = [
        {"input": "Q1", "expected_output": "hello world", "actual_output": "hello world"},
        {"input": "Q2", "expected_output": "foo bar baz", "actual_output": "foo bar baz"},
    ]
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(data))
    return path


class TestRunCommand:
    def test_basic_run(self, tmp_path: Path) -> None:
        ds = _make_dataset(tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "-d", str(ds), "-m", "bleu,rouge"])
        assert result.exit_code == 0
        assert "bleu" in result.output.lower()
        assert "rouge" in result.output.lower()

    def test_run_with_output_json(self, tmp_path: Path) -> None:
        ds = _make_dataset(tmp_path)
        out = tmp_path / "results.json"
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "-d", str(ds), "-m", "rouge", "-o", str(out)])
        assert result.exit_code == 0
        assert out.exists()
        data = json.loads(out.read_text())
        assert "run" in data

    def test_run_with_output_csv(self, tmp_path: Path) -> None:
        ds = _make_dataset(tmp_path)
        out = tmp_path / "results.csv"
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "-d", str(ds), "-m", "bleu", "-o", str(out)])
        assert result.exit_code == 0
        assert out.exists()
        assert "score" in out.read_text()

    def test_run_missing_dataset(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "-d", "/nonexistent.json", "-m", "bleu"])
        assert result.exit_code != 0

    def test_run_invalid_metric(self, tmp_path: Path) -> None:
        ds = _make_dataset(tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "-d", str(ds), "-m", "nonexistent"])
        assert result.exit_code != 0

    def test_run_save_baseline(self, tmp_path: Path) -> None:
        ds = _make_dataset(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            cli, ["run", "-d", str(ds), "-m", "rouge", "--save-baseline", "test_bl"]
        )
        assert result.exit_code == 0
        assert "baseline" in result.output.lower()


class TestResultsCommand:
    def test_results_empty(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["results"])
        assert result.exit_code == 0

    def test_results_json_format(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["results", "--format", "json"])
        assert result.exit_code == 0


class TestVersionFlag:
    def test_version(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output
