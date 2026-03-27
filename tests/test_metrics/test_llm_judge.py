"""Tests for LLM-as-judge metric."""

import json
from unittest.mock import MagicMock, patch

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

    def test_json_in_plain_code_block(self) -> None:
        response = '```\n{"accuracy": 8, "overall": 6, "reasoning": "ok"}\n```'
        result = _parse_judge_response(response)
        assert result["overall"] == 6

    def test_json_with_surrounding_text(self) -> None:
        response = 'Here is my evaluation:\n{"accuracy": 9, "overall": 8, "reasoning": "great"}\nThat is all.'
        result = _parse_judge_response(response)
        assert result["overall"] == 8

    def test_unparseable(self) -> None:
        with pytest.raises(ValueError, match="Could not parse"):
            _parse_judge_response("This is not JSON at all")

    def test_empty_string(self) -> None:
        with pytest.raises(ValueError, match="Could not parse"):
            _parse_judge_response("")

    def test_malformed_json(self) -> None:
        with pytest.raises(ValueError, match="Could not parse"):
            _parse_judge_response("{accuracy: 8, overall: 7}")


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
        assert "{hypothesis}" in metric.prompt_template

    def test_default_model(self) -> None:
        metric = LLMJudgeMetric(api_key="fake")
        assert metric.model == "gpt-4o-mini"

    def test_custom_model(self) -> None:
        metric = LLMJudgeMetric(api_key="fake", model="gpt-4o")
        assert metric.model == "gpt-4o"

    def test_default_temperature(self) -> None:
        metric = LLMJudgeMetric(api_key="fake")
        assert metric.temperature == 0.0

    def test_compute_with_mock_client(self) -> None:
        metric = LLMJudgeMetric(api_key="test-key")

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "accuracy": 9, "completeness": 8, "clarity": 7, "overall": 8,
            "reasoning": "solid output",
        })

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        metric._client = mock_client

        result = metric.compute("expected answer", "model answer")

        # overall=8 -> normalized = (8-1)/9 = 7/9
        assert result.score == pytest.approx(7 / 9, abs=0.01)
        assert result.details["model"] == "gpt-4o-mini"
        assert result.details["raw_scores"]["overall"] == 8
        assert result.details["reasoning"] == "solid output"

        # Verify the client was called with correct args
        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args
        assert call_kwargs.kwargs["model"] == "gpt-4o-mini"
        assert call_kwargs.kwargs["temperature"] == 0.0

    def test_score_normalization_boundaries(self) -> None:
        """Test that score normalization clamps correctly at edges."""
        metric = LLMJudgeMetric(api_key="test-key")

        for raw, expected in [(1, 0.0), (10, 1.0), (5, 4 / 9)]:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = json.dumps({"overall": raw, "reasoning": ""})

            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            metric._client = mock_client

            result = metric.compute("ref", "hyp")
            assert result.score == pytest.approx(expected, abs=0.01), f"raw={raw}"

    def test_score_clamped_when_out_of_range(self) -> None:
        """Overall values outside 1-10 are clamped before normalization."""
        metric = LLMJudgeMetric(api_key="test-key")

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({"overall": 15, "reasoning": ""})

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        metric._client = mock_client

        result = metric.compute("ref", "hyp")
        # Clamped to 10, normalized to 1.0
        assert result.score == pytest.approx(1.0)

    def test_invalid_prompt_template(self) -> None:
        metric = LLMJudgeMetric(api_key="test-key", prompt_template="No placeholders here")
        mock_client = MagicMock()
        metric._client = mock_client
        # Template without {reference}/{hypothesis} won't raise on format if no braces
        # but a template with {unknown} would
        metric2 = LLMJudgeMetric(api_key="test-key", prompt_template="Rate: {unknown}")
        metric2._client = mock_client
        with pytest.raises(ValueError, match="Failed to format"):
            metric2.compute("ref", "hyp")

    @patch.dict("os.environ", {"OPENAI_API_KEY": "env-key", "ML_EVAL_JUDGE_MODEL": "gpt-4o"})
    def test_env_var_fallbacks(self) -> None:
        metric = LLMJudgeMetric()
        assert metric.api_key == "env-key"
        assert metric.model == "gpt-4o"
