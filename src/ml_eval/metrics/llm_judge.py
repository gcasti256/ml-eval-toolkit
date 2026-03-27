"""LLM-as-judge evaluation via OpenAI API."""

from __future__ import annotations

import json
import os
import re
from typing import Any

from ml_eval.metrics.base import BaseMetric, MetricResult

DEFAULT_JUDGE_PROMPT = """You are an expert evaluator. Score the following model output against the reference.

## Reference (Expected Output)
{reference}

## Model Output (To Evaluate)
{hypothesis}

## Instructions
Evaluate the model output on a scale of 1-10 across these dimensions:
1. **Accuracy**: How factually correct is the output compared to the reference?
2. **Completeness**: Does it cover all key points from the reference?
3. **Clarity**: Is the output well-structured and easy to understand?

Respond with ONLY a JSON object in this exact format:
{{"accuracy": <1-10>, "completeness": <1-10>, "clarity": <1-10>, "overall": <1-10>, "reasoning": "<brief explanation>"}}
"""


def _parse_judge_response(response_text: str) -> dict[str, Any]:
    """Extract structured scores from the LLM judge response."""
    # Try direct JSON parse
    try:
        return dict(json.loads(response_text))
    except json.JSONDecodeError:
        pass

    # Try extracting JSON from markdown code block
    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", response_text)
    if json_match:
        try:
            return dict(json.loads(json_match.group(1).strip()))
        except json.JSONDecodeError:
            pass

    # Try finding any JSON-like object
    json_match = re.search(r"\{[^{}]*\}", response_text)
    if json_match:
        try:
            return dict(json.loads(json_match.group(0)))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse judge response: {response_text[:200]}")


class LLMJudgeMetric(BaseMetric):
    """Use an LLM as an evaluator/judge.

    Args:
        model: OpenAI model name for the judge (default from env or gpt-4o-mini).
        prompt_template: Custom prompt template. Must contain {reference} and {hypothesis} placeholders.
        api_key: OpenAI API key. Falls back to OPENAI_API_KEY env var.
        temperature: Sampling temperature for the judge model.
    """

    def __init__(
        self,
        model: str = "",
        prompt_template: str = "",
        api_key: str = "",
        temperature: float = 0.0,
        **_: Any,
    ) -> None:
        self.model = model or os.environ.get("ML_EVAL_JUDGE_MODEL", "gpt-4o-mini")
        self.prompt_template = prompt_template or DEFAULT_JUDGE_PROMPT
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.temperature = temperature
        self._client: Any = None

    @property
    def name(self) -> str:
        return "llm_judge"

    def _get_client(self) -> Any:
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self.api_key)
        return self._client

    def compute(self, reference: str, hypothesis: str) -> MetricResult:
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required for LLM judge. "
                "Set OPENAI_API_KEY env var or pass api_key parameter."
            )

        try:
            prompt = self.prompt_template.format(reference=reference, hypothesis=hypothesis)
        except (KeyError, IndexError) as e:
            raise ValueError(f"Failed to format judge prompt template: {e}") from e
        client = self._get_client()

        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )

        response_text = response.choices[0].message.content or ""
        scores = _parse_judge_response(response_text)

        # Normalize overall score from 1-10 to 0-1
        raw_overall = scores.get("overall", 5)
        overall = max(1, min(10, float(raw_overall)))
        normalized = (overall - 1) / 9

        return MetricResult(
            score=normalized,
            details={
                "model": self.model,
                "raw_scores": scores,
                "reasoning": scores.get("reasoning", ""),
                "raw_response": response_text,
            },
        )
