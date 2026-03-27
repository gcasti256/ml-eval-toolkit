"""Evaluation orchestration, comparison, and regression testing."""

from ml_eval.evaluation.comparison import compare_configs
from ml_eval.evaluation.regression import check_regression
from ml_eval.evaluation.runner import EvalRunner

__all__ = ["EvalRunner", "check_regression", "compare_configs"]
