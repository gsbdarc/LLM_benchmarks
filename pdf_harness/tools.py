"""Trusted built-in tool catalog used by task-scoped MCP/agent runs.

The UI can select these tools, but it cannot upload code or issue arbitrary database
queries. Each tool is registered with a small JSON-compatible contract.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

from jsonschema import Draft202012Validator


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    fn: Callable[..., Any]
    stages: frozenset[str]


_TOOLS: dict[str, ToolDefinition] = {}


def builtin_tool(name: str, description: str, stages: tuple[str, ...]):
    def register(fn: Callable[..., Any]) -> Callable[..., Any]:
        if name in _TOOLS:
            raise ValueError(f"duplicate built-in tool {name}")
        _TOOLS[name] = ToolDefinition(name, description, fn, frozenset(stages))
        return fn
    return register


def catalog() -> dict[str, ToolDefinition]:
    return dict(_TOOLS)


def validate_allowlist(names: list[str], stage: str | None = None) -> list[str]:
    errors: list[str] = []
    for name in names:
        tool = _TOOLS.get(name)
        if not tool:
            errors.append(f"unknown built-in tool: {name}")
        elif stage and stage not in tool.stages:
            errors.append(f"tool {name} is not allowed for stage {stage}")
    return errors


def invoke(name: str, allowlist: list[str], **arguments: Any) -> Any:
    if name not in allowlist:
        raise PermissionError(f"tool {name!r} is not enabled for this task")
    tool = _TOOLS.get(name)
    if not tool:
        raise KeyError(f"unknown built-in tool {name!r}")
    return tool.fn(**arguments)


@builtin_tool("validate_schema", "Validate a structured result against the task JSON schema.", ("extraction", "aggregation"))
def validate_schema(value: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    errors = sorted(Draft202012Validator(schema).iter_errors(value), key=lambda e: list(e.path))
    messages = [f"{'.'.join(map(str, e.path)) or '$'}: {e.message}" for e in errors]
    return {"valid": not messages, "errors": messages}


@builtin_tool("normalize_text", "Normalize whitespace and optional case for deterministic comparison.", ("extraction", "score", "judge"))
def normalize_text(value: Any, lowercase: bool = False) -> str:
    text = re.sub(r"\s+", " ", "" if value is None else str(value)).strip()
    return text.lower() if lowercase else text


@builtin_tool("parse_date", "Parse a date into ISO YYYY-MM-DD where possible.", ("extraction", "score", "judge"))
def parse_date(value: Any) -> str | None:
    if value in (None, ""):
        return None
    text = normalize_text(value)
    for fmt in (
        "%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%B %d, %Y", "%b %d, %Y",
        "%B %d %Y", "%b %d %Y",
    ):
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            continue
    return None


@builtin_tool("coerce_number", "Coerce formatted numeric text to a number.", ("extraction", "score", "judge"))
def coerce_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    cleaned = re.sub(r"[$,%\s,]", "", str(value))
    try:
        return float(cleaned)
    except ValueError:
        return None


@builtin_tool("deduplicate_list", "Deduplicate a list while preserving order.", ("extraction", "aggregation", "score"))
def deduplicate_list(values: list[Any] | None) -> list[Any]:
    result: list[Any] = []
    for value in values or []:
        if value not in result:
            result.append(value)
    return result
