"""Task-scoped MCP server exposing only reviewed application-owned functions."""

from __future__ import annotations

import os

from mcp.server.fastmcp import FastMCP

from .bootstrap import build_context
from .models import new_id, utcnow
from .tools import coerce_number, deduplicate_list, normalize_text, parse_date, validate_schema

mcp = FastMCP("PDF Extraction Harness", host=os.getenv("MCP_HOST", "127.0.0.1"), port=int(os.getenv("MCP_PORT", "8000")))
_context = None


def _ctx():
    global _context
    _context = _context or build_context()
    return _context


def _scoped_work(work_item_id: str):
    work = _ctx().repo.get("work_items", work_item_id)
    if not work:
        raise ValueError("work item not found")
    run_scope = os.getenv("HARNESS_RUN_ID")
    if not run_scope or work.get("run_id") != run_scope:
        raise PermissionError("work item is outside this MCP server's run scope")
    return work


@mcp.tool()
def get_work_context(work_item_id: str) -> dict:
    """Return sanitized task, page, model, and prompt metadata for one in-scope work item."""
    work = _scoped_work(work_item_id)
    run = _ctx().repo.get("runs", work["run_id"])
    document = _ctx().repo.get("documents", work["document_id"])
    return {
        "work_item_id": work_item_id,
        "run_id": work["run_id"],
        "stage": work["stage"],
        "document": {"id": document["id"], "name": document["name"], "page_number": work.get("page_number")},
        "task_spec_version_id": run["snapshot"]["task_spec_version_id"],
        "model_profile_version_id": work.get("model_profile_version_id"),
        "prompt_version_id": work.get("prompt_version_id"),
    }


@mcp.tool()
def validate_structured_result(work_item_id: str, value: dict) -> dict:
    """Validate a proposed result against the immutable schema for this in-scope run."""
    work = _scoped_work(work_item_id)
    run = _ctx().repo.get("runs", work["run_id"])
    spec = _ctx().repo.get("task_specs", run["snapshot"]["task_spec_version_id"])
    return validate_schema(value, spec["json_schema"])


@mcp.tool()
def normalize_text_value(value: str, lowercase: bool = False) -> str:
    """Normalize whitespace and optionally lowercase text."""
    return normalize_text(value, lowercase)


@mcp.tool()
def parse_date_value(value: str) -> str | None:
    """Parse common date formats into ISO YYYY-MM-DD."""
    return parse_date(value)


@mcp.tool()
def coerce_numeric_value(value: str) -> float | None:
    """Coerce currency, percent, and comma-formatted numeric text."""
    return coerce_number(value)


@mcp.tool()
def deduplicate_values(values: list) -> list:
    """Deduplicate values while preserving their first-seen order."""
    return deduplicate_list(values)


@mcp.tool()
def save_review_decision(work_item_id: str, extraction_id: str, decision: str, comment: str = "") -> dict:
    """Save an approve/reject/needs_review decision for an in-scope extraction."""
    work = _scoped_work(work_item_id)
    if decision not in {"approved", "rejected", "needs_review"}:
        raise ValueError("decision must be approved, rejected, or needs_review")
    extraction = _ctx().repo.get("extractions", extraction_id)
    if not extraction or extraction.get("work_item_id") != work_item_id:
        raise ValueError("extraction does not belong to the in-scope work item")
    event = {
        "id": new_id(), "project_id": work["project_id"], "run_id": work["run_id"],
        "work_item_id": work_item_id, "extraction_id": extraction_id,
        "action": "extraction_review", "decision": decision,
        "comment": comment[:2000], "created_at": utcnow(),
    }
    _ctx().repo.put("audit_events", event)
    return {"saved": True, "audit_event_id": event["id"], "decision": decision}


def main() -> None:
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
