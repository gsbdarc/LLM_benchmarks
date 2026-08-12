"""OpenAI-compatible multimodal extraction adapter."""

from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass
from typing import Any, Protocol

from openai import OpenAI

from .models import ModelProfileVersion, PromptVersion


@dataclass
class LLMResult:
    output: dict[str, Any] | None
    raw_response: str
    input_tokens: int
    output_tokens: int
    latency_seconds: float


class ExtractionClient(Protocol):
    def test_connection(self, profile: ModelProfileVersion, api_key: str) -> tuple[bool, str]: ...
    def extract(
        self, profile: ModelProfileVersion, prompt: PromptVersion, api_key: str,
        image: bytes, image_content_type: str, schema: dict[str, Any],
        variables: dict[str, Any],
    ) -> LLMResult: ...
    def aggregate(
        self, profile: ModelProfileVersion, prompt: PromptVersion, api_key: str,
        page_outputs: list[dict[str, Any]], schema: dict[str, Any], variables: dict[str, Any],
    ) -> LLMResult: ...
    def judge(
        self, profile: ModelProfileVersion, prompt: PromptVersion, api_key: str,
        predicted: dict[str, Any], expected: dict[str, Any],
        deterministic_scores: dict[str, Any], variables: dict[str, Any],
    ) -> LLMResult: ...
    def repair(
        self, profile: ModelProfileVersion, api_key: str, raw_response: str,
        schema: dict[str, Any], variables: dict[str, Any],
    ) -> LLMResult: ...


def _json_content(content: str) -> dict[str, Any]:
    cleaned = content.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[len("```json"):]
    elif cleaned.startswith("```"):
        cleaned = cleaned[len("```"):]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    value = json.loads(cleaned.strip())
    if not isinstance(value, dict):
        raise ValueError("model response must be a JSON object")
    return value


class OpenAICompatibleExtractionClient:
    def _client(self, profile: ModelProfileVersion, api_key: str) -> OpenAI:
        return OpenAI(base_url=profile.base_url.rstrip("/"), api_key=api_key, max_retries=0, timeout=600)

    def test_connection(self, profile: ModelProfileVersion, api_key: str) -> tuple[bool, str]:
        try:
            response = self._client(profile, api_key).chat.completions.create(
                model=profile.model_id,
                messages=[{"role": "user", "content": "Reply with OK."}],
                max_tokens=5,
                temperature=0,
            )
            if response.choices:
                return True, f"{profile.model_id} responded successfully"
            return False, f"{profile.model_id} returned no choices"
        except Exception as exc:  # noqa: BLE001
            return False, f"connection failed: {type(exc).__name__}: {str(exc)[:160]}"

    def extract(
        self, profile: ModelProfileVersion, prompt: PromptVersion, api_key: str,
        image: bytes, image_content_type: str, schema: dict[str, Any],
        variables: dict[str, Any],
    ) -> LLMResult:
        user_text = prompt.user_template.format_map(variables)
        data_uri = f"data:{image_content_type};base64,{base64.b64encode(image).decode()}"
        params = dict(profile.request_parameters)
        params.pop("model", None)
        started = time.monotonic()
        response = self._client(profile, api_key).chat.completions.create(
            model=profile.model_id,
            messages=[
                {"role": "system", "content": prompt.system_template},
                {"role": "user", "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ]},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "extraction_result", "strict": True, "schema": schema},
            },
            **params,
        )
        latency = time.monotonic() - started
        raw = response.choices[0].message.content or ""
        usage = response.usage
        try:
            parsed = _json_content(raw)
        except (json.JSONDecodeError, ValueError):
            parsed = None
        return LLMResult(
            output=parsed,
            raw_response=raw,
            input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            latency_seconds=latency,
        )

    def repair(
        self, profile: ModelProfileVersion, api_key: str, raw_response: str,
        schema: dict[str, Any], variables: dict[str, Any],
    ) -> LLMResult:
        started = time.monotonic()
        response = self._client(profile, api_key).chat.completions.create(
            model=profile.model_id,
            messages=[
                {"role": "system", "content": "Repair the supplied extraction into valid JSON matching the schema. Do not add unsupported facts."},
                {"role": "user", "content": f"Document: {variables.get('document_name')}, page {variables.get('page_number')}\nInvalid extraction:\n{raw_response[:20000]}"},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "repaired_extraction", "strict": True, "schema": schema},
            },
            temperature=0,
        )
        latency = time.monotonic() - started
        raw = response.choices[0].message.content or ""
        usage = response.usage
        try:
            parsed = _json_content(raw)
        except (json.JSONDecodeError, ValueError):
            parsed = None
        return LLMResult(
            output=parsed, raw_response=raw,
            input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            latency_seconds=latency,
        )

    def aggregate(
        self, profile: ModelProfileVersion, prompt: PromptVersion, api_key: str,
        page_outputs: list[dict[str, Any]], schema: dict[str, Any], variables: dict[str, Any],
    ) -> LLMResult:
        rendered_pages = json.dumps(page_outputs, ensure_ascii=False, default=str)
        prompt_variables = {**variables, "page_outputs": rendered_pages}
        user_text = prompt.user_template.format_map(prompt_variables)
        params = dict(profile.request_parameters)
        params.pop("model", None)
        started = time.monotonic()
        response = self._client(profile, api_key).chat.completions.create(
            model=profile.model_id,
            messages=[
                {"role": "system", "content": prompt.system_template},
                {"role": "user", "content": user_text},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "document_result", "strict": True, "schema": schema},
            },
            **params,
        )
        latency = time.monotonic() - started
        raw = response.choices[0].message.content or ""
        usage = response.usage
        try:
            parsed = _json_content(raw)
        except (json.JSONDecodeError, ValueError):
            parsed = None
        return LLMResult(
            output=parsed, raw_response=raw,
            input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            latency_seconds=latency,
        )

    def judge(
        self, profile: ModelProfileVersion, prompt: PromptVersion, api_key: str,
        predicted: dict[str, Any], expected: dict[str, Any],
        deterministic_scores: dict[str, Any], variables: dict[str, Any],
    ) -> LLMResult:
        judge_schema = {
            "type": "object",
            "properties": {
                "score": {"type": "number", "minimum": 0, "maximum": 1},
                "rationale": {"type": "string"},
                "needs_review": {"type": "boolean"},
            },
            "required": ["score", "rationale", "needs_review"],
            "additionalProperties": False,
        }
        prompt_variables = {
            **variables,
            "predicted": json.dumps(predicted, ensure_ascii=False, default=str),
            "expected": json.dumps(expected, ensure_ascii=False, default=str),
            "deterministic_scores": json.dumps(deterministic_scores, ensure_ascii=False, default=str),
        }
        started = time.monotonic()
        response = self._client(profile, api_key).chat.completions.create(
            model=profile.model_id,
            messages=[
                {"role": "system", "content": prompt.system_template},
                {"role": "user", "content": prompt.user_template.format_map(prompt_variables)},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "judge_result", "strict": True, "schema": judge_schema},
            },
            temperature=0,
        )
        latency = time.monotonic() - started
        raw = response.choices[0].message.content or ""
        usage = response.usage
        try:
            parsed = _json_content(raw)
        except (json.JSONDecodeError, ValueError):
            parsed = None
        return LLMResult(
            output=parsed, raw_response=raw,
            input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            latency_seconds=latency,
        )


def estimated_cost(profile: ModelProfileVersion, input_tokens: int, output_tokens: int) -> float:
    input_price = profile.input_price_per_million or 0
    output_price = profile.output_price_per_million or 0
    return (input_tokens * input_price + output_tokens * output_price) / 1_000_000
