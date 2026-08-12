"""Deterministic typed benchmark metrics; LLM judging is a separate optional layer."""

from __future__ import annotations

from collections import Counter
from difflib import SequenceMatcher
from typing import Any

from .models import FieldRule
from .tools import coerce_number, normalize_text, parse_date


def _exact(predicted: Any, expected: Any) -> dict[str, Any]:
    return {"score": float(predicted == expected), "metric": "exact"}


def _normalized_exact(predicted: Any, expected: Any) -> dict[str, Any]:
    p = normalize_text(predicted, lowercase=True)
    e = normalize_text(expected, lowercase=True)
    return {"score": float(p == e), "metric": "normalized_exact", "predicted_normalized": p, "expected_normalized": e}


def _text(predicted: Any, expected: Any) -> dict[str, Any]:
    p = normalize_text(predicted, lowercase=True)
    e = normalize_text(expected, lowercase=True)
    return {"score": SequenceMatcher(None, p, e).ratio(), "metric": "text_similarity"}


def _numeric(predicted: Any, expected: Any, tolerance: float) -> dict[str, Any]:
    p, e = coerce_number(predicted), coerce_number(expected)
    if p is None or e is None:
        return {"score": 0.0, "metric": "numeric", "parse_error": True}
    delta = abs(p - e)
    return {"score": float(delta <= tolerance), "metric": "numeric", "absolute_error": delta, "tolerance": tolerance}


def _date(predicted: Any, expected: Any) -> dict[str, Any]:
    p, e = parse_date(predicted), parse_date(expected)
    return {"score": float(p is not None and p == e), "metric": "date", "predicted_parsed": p, "expected_parsed": e}


def _boolean(predicted: Any, expected: Any) -> dict[str, Any]:
    def coerce(v: Any) -> bool | None:
        if isinstance(v, bool):
            return v
        if isinstance(v, str) and v.strip().lower() in {"true", "yes", "1"}:
            return True
        if isinstance(v, str) and v.strip().lower() in {"false", "no", "0"}:
            return False
        return None
    p, e = coerce(predicted), coerce(expected)
    return {"score": float(p is not None and p == e), "metric": "boolean", "predicted_parsed": p, "expected_parsed": e}


def _list(predicted: Any, expected: Any) -> dict[str, Any]:
    if not isinstance(predicted, list) or not isinstance(expected, list):
        return {"score": 0.0, "metric": "list", "type_error": True}
    p = Counter(normalize_text(v, lowercase=True) for v in predicted)
    e = Counter(normalize_text(v, lowercase=True) for v in expected)
    overlap = sum((p & e).values())
    precision = overlap / sum(p.values()) if p else float(not e)
    recall = overlap / sum(e.values()) if e else float(not p)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"score": f1, "metric": "list", "precision": precision, "recall": recall, "f1": f1}


def infer_metric(schema: dict[str, Any], rule: FieldRule) -> str:
    if rule.scoring != "auto":
        return rule.scoring
    kind = schema.get("type")
    if isinstance(kind, list):
        kind = next((v for v in kind if v != "null"), "string")
    fmt = schema.get("format")
    if fmt == "date":
        return "date"
    return {
        "number": "numeric", "integer": "numeric", "boolean": "boolean",
        "array": "list", "object": "exact",
    }.get(kind, "normalized_exact")


def score_field(predicted: Any, expected: Any, schema: dict[str, Any], rule: FieldRule | None = None) -> dict[str, Any]:
    rule = rule or FieldRule()
    metric = infer_metric(schema, rule)
    if metric == "exact":
        return _exact(predicted, expected)
    if metric == "normalized_exact":
        return _normalized_exact(predicted, expected)
    if metric == "text_similarity":
        return _text(predicted, expected)
    if metric == "numeric":
        return _numeric(predicted, expected, rule.numeric_tolerance or 0.0)
    if metric == "date":
        return _date(predicted, expected)
    if metric == "boolean":
        return _boolean(predicted, expected)
    if metric == "list":
        return _list(predicted, expected)
    raise ValueError(f"unsupported scoring metric {metric}")


def score_record(
    predicted: dict[str, Any], expected: dict[str, Any], json_schema: dict[str, Any],
    field_rules: dict[str, FieldRule] | None = None,
) -> dict[str, Any]:
    properties = json_schema.get("properties", {})
    rules = field_rules or {}
    fields = {
        field: score_field(predicted.get(field), expected.get(field), schema, rules.get(field))
        for field, schema in properties.items()
        if field in expected
    }
    scores = [v["score"] for v in fields.values()]
    required = set(json_schema.get("required", []))
    present = sum(1 for field in required if predicted.get(field) not in (None, "", []))
    return {
        "overall_score": sum(scores) / len(scores) if scores else None,
        "required_field_coverage": present / len(required) if required else 1.0,
        "fields": fields,
    }
