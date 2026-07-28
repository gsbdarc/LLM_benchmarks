"""
server.py — FastMCP server exposing metric-evaluation tools over HTTP.

Thin "waiter" layer over tools.py (the "kitchen"), in the same shape as
server_http.py is over redivis_tools.py. Three data tools (list / fetch / save)
plus three composite type-tools, one per field data-shape.

The tool *descriptions* are deliberately written to teach the model WHEN each
composite applies, since the agent is expected to classify each field's data
shape (free-form text / single extracted value / list of items) from the values
alone and route to the matching type-tool. The descriptions never name the
benchmark's declared type — that would defeat the point. Each composite runs the
full per-type scoring internally (the seven underlying metrics are private).
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

from mcp.server.fastmcp import FastMCP

from .tools import (
    fetch_evaluable_output,
    list_pending_outputs,
    save_agentic_evaluation,
    composite_raw_string as _composite_raw_string,
    composite_extracted_string as _composite_extracted_string,
    composite_list as _composite_list,
)

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format="%(asctime)s %(levelname)s %(message)s",
)

HOST = os.getenv("MCP_HOST", "127.0.0.1")
PORT = int(os.getenv("MCP_PORT", "8000"))

mcp = FastMCP(
    "metric-eval-http",
    json_response=True,
    host=HOST,
    port=PORT,
)


def _composite_tool(fn, predicted, expected, field_type: str) -> dict:
    """Run a per-type composite and flatten its result.

    The composite helper returns {composite_score, components, field_type}; we
    flatten the components to the top level so (a) the agent sees every number
    and (b) the score-consistency check finds every saved score in a tool result.
    Any unexpected error is surfaced as data, never crashing the tool.
    """
    try:
        res = fn(predicted, expected)
    except Exception as e:  # noqa: BLE001 — surface as data, never crash the tool
        logging.exception("composite %s failed", field_type)
        return {"error": f"metric error: {e}"}
    return {
        "field_type": res["field_type"],
        "composite_score": res["composite_score"],
        **res["components"],
    }


# ---------------------------------------------------------------------------
#  Data tools — read from / write to MongoDB
# ---------------------------------------------------------------------------

@mcp.tool()
def list_outputs(benchmark_id: str | None = None, limit: int = 20) -> dict:
    """
    List model outputs that are ready to be evaluated.

    Returns (task_id, run_id) identifier pairs for processed outputs of evaluable
    benchmarks. A single task_id has multiple runs (run_id), so always carry both
    forward. Optionally filter to one benchmark_id. Use this first to discover
    what to evaluate, then call get_task_output for each item.
    """
    try:
        return list_pending_outputs(benchmark_id=benchmark_id, limit=limit)
    except Exception as e:  # noqa: BLE001
        logging.exception("list_outputs failed")
        return {"error": f"db error: {e}", "count": 0, "outputs": []}


@mcp.tool()
def get_task_output(task_id: str, run_id: int | None = None) -> dict:
    """
    Fetch one model output paired with its ground truth, ready to judge.

    Returns, per field, the `predicted` value (what the model produced) and the
    `expected` value (ground truth). YOU must decide the field's DATA SHAPE by
    inspecting those values and route to exactly one composite type-tool — the
    correct type is not provided.

    Guidance for classifying a field (pick ONE type-tool per field):
      * Free-form / multi-word text (a raw line as printed) -> evaluate_raw_string
      * A single short extracted value that may legitimately be absent
        (null/empty) -> evaluate_extracted_string
      * A list / collection of items -> evaluate_list

    Each type-tool runs the full per-type scoring internally and returns a
    "composite_score". Pass run_id to target a specific run; omit it to get the
    first matching run.
    """
    try:
        return fetch_evaluable_output(task_id, run_id=run_id)
    except Exception as e:  # noqa: BLE001
        logging.exception("get_task_output failed")
        return {"error": f"db error: {e}", "fields": []}


@mcp.tool()
def save_evaluation(
    task_id: str,
    benchmark_id: str,
    model_id: str,
    run_id: int,
    image_id: Any,
    field_evaluations: list,
    eval_id: int | None = None,
    git_commit: str | None = None,
) -> dict:
    """
    Persist your evaluation of one model output to the agentic_evaluations
    collection. Keyed on (eval_id, git_commit) so results are versioned by code.

    Leave `eval_id` and `git_commit` unset — the client stamps them automatically to
    correlate this verdict with its run-metrics row and code version; any value you
    pass is overwritten.

    Pass back the identifiers exactly as returned by get_task_output. Each entry
    in field_evaluations records one field's verdict, e.g.:
        {"field": "first_program", "field_type": "raw_string",
         "metric": "evaluate_raw_string",
         "scores": {"composite_score": 0.67, "word_iou": 0.67},
         "rationale": "free-form OCR text"}
    Set "metric" to the type-tool you called and "field_type" to the shape you
    inferred. Include the dict that type-tool returned under "scores", and a short
    rationale for why that data shape fits.
    """
    try:
        return save_agentic_evaluation(
            task_id=task_id,
            benchmark_id=benchmark_id,
            model_id=model_id,
            run_id=run_id,
            image_id=image_id,
            field_evaluations=field_evaluations,
            eval_id=eval_id,
            git_commit=git_commit,
        )
    except Exception as e:  # noqa: BLE001
        logging.exception("save_evaluation failed")
        return {"error": f"db error: {e}", "saved": False}


# ---------------------------------------------------------------------------
#  Composite type-tools — one per field data-shape. Each runs the full per-type
#  scoring internally (the seven underlying metrics are private) and returns a
#  "composite_score" plus its component sub-scores.
# ---------------------------------------------------------------------------

@mcp.tool()
def evaluate_raw_string(predicted: str | None = None, expected: str | None = None) -> dict:
    """
    Composite score for a FREE-FORM / multi-word text field — a raw line as
    printed, where the right words matter but exact order and punctuation do not
    (e.g. a raw channel/program line).

    Use this when the values are prose-like strings: not a single short token and
    not a list. Returns "composite_score" (word-IoU based, in [0,1]; both empty =
    1.0) plus the "word_iou" sub-score.
    """
    return _composite_tool(_composite_raw_string, predicted, expected, "raw_string")


@mcp.tool()
def evaluate_extracted_string(predicted: str | None = None, expected: str | None = None) -> dict:
    """
    Composite score for a SINGLE extracted value that may legitimately be absent
    (null/empty) — e.g. one channel number or a station name.

    First scores presence (did the model correctly decide whether a value
    exists?); when present, scores content with the better of edit-distance and
    character-F1, and the presence score gates the content score. Use this for a
    lone scalar-ish string — NOT prose, NOT a list. Represent an absent value as
    "" or null. Returns "composite_score" plus presence/content sub-scores.
    """
    return _composite_tool(_composite_extracted_string, predicted, expected, "extracted_string")


@mcp.tool()
def evaluate_list(predicted: list | None = None, expected: list | None = None) -> dict:
    """
    Composite score for a LIST / collection of items — e.g. all channels or
    ordered time slots.

    Internally takes the best of order-independent set-F1, order-sensitive LCS,
    and substring-aware inclusion (a ground-truth item embedded in a longer
    predicted string), so it credits correct items whether or not order matters.
    Use this whenever the values are collections of items. Returns
    "composite_score" plus the list sub-scores.
    """
    return _composite_tool(_composite_list, predicted, expected, "list")


if __name__ == "__main__":
    logging.info("Starting Metric-Eval MCP server on http://%s:%s/mcp", HOST, PORT)
    mcp.run(transport="streamable-http")
