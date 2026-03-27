"""Tests for LLM-as-judge metric."""

import json

import pytest

from ml_eval.metrics.llm_judge import LLMJudgeMetric, _parse_judge_response


class TestParseJudgeResponse:
    def test_valid_json(self) -> None:
        response = json.dumps({"accuracy": 8, "completeness": 7, "clarity": 9, "overall": 8, "reasoning": "good"})
        result = _parse_judge_response(response)
        assert result["overall"] == 8
        assert result["reasoning"] == "good"

    def test_json_in_code_block(self) -> None:
        response = '```json\n{"accuracy": 8, "overall": 7, "reasoning": "ok"}\n```'
        result = _parse_judge_response(response)
        assert result["overall"] == 7

    def test_json_with_surrounding_text(self) -> None:
        response = 'Here is my evaluation:\n{"accuracy": 9, "overall": 8, "reasoning": "great"}\nThat is all.'
        result = _parse_judge_response(response)
        assert result["overall"] == 8

    def test_unparseable(self) -> None:
        with pytest.raises(ValueError, match="Could not parse"):
            _parse_judge_response("This is not JSON at all")


class TestLLMJudgeMetric:
    def test_name(self) -> None:
        metric = LLMJudgeMetric(api_key="fake")
        assert metric.name == "llm_judge"

    def test_missing_api_key(self) -> None:
        metric = LLMJudgeMetric(api_key="")
        with pytest.raises(ValueError, match="API key required"):
            metric.compute("reference", "hypothesis")

    def test_custom_template(self) -> None:
        template = "Rate this: ref={reference} hyp={hypothesis}"
        metric = LLMJudgeMetric(api_key="fake", prompt_template=template)
        assert "{reference}" in metric.prompt_template

    def test_default_model(self) -> None:
        metric = LLMJudgeMetric(api_key="fake")
        assert metric.model == "gpt-4o-mini"

    def test_custom_model(self) -> None:
        metric = LLMJudgeMetric(api_key="fake", model="gpt-4o")
        assert metric.model == "gpt-4o"
