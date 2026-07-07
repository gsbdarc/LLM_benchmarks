"""
metric_tools.py — Business-logic layer for the agentic metric-evaluation MCP server.

Mirrors the role redivis_tools.py plays for the Redivis server: it owns MongoDB
access, input validation, and the metric calculators. The FastMCP server
(metric_server.py) is a thin wrapper that exposes these as tools.

Design notes for whoever picks this up:
  * The seven metric functions are imported from scripts/evaluator.py — we do NOT
    reimplement them. There is one source of truth for "what word_iou means".
  * fetch_evaluable_output deliberately strips the benchmark spec's declared
    `type` and `metrics`. The whole point of the agentic flow is that the LLM
    looks at the actual predicted/expected values and decides which metric fits.
    If we handed it the answer key, there would be no reasoning to do.
  * Output shapes seen in llm_outputs: a single-field benchmark stores a bare
    value (e.g. "News 8 Austin"); a multi-field/structured benchmark stores a
    dict (e.g. {"first_channel_raw": "...", "first_channel_numbers": ["2"]}).
    Field *values* are always a string or a list — never a scalar number or a
    nested object — which is why the metric layer only has to handle those two.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.server_api import ServerApi

# ---------------------------------------------------------------------------
#  Reuse the tested metric functions from the deterministic pipeline.
#  scripts/evaluator.py is the single source of truth for metric definitions.
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR / "scripts"))

from evaluator import (  # noqa: E402  (import after sys.path injection)
    # The seven metric functions are now PRIVATE helpers behind the composites;
    # kept imported for METRIC_REGISTRY / any non-agentic callers.
    word_iou,
    null_accuracy,
    levenshtein_similarity,
    char_f1,
    set_f1,
    sequence_lcs,
    set_inclusion,
    # Per-type composites — the agent-facing surface is now these three.
    composite_raw_string,
    composite_extracted_string,
    composite_list,
    _resolve_gt_value,
)

# field_type → composite function. The agent classifies a field's data shape and
# the server exposes one composite tool per type (evaluate_<type>).
COMPOSITE_REGISTRY = {
    "raw_string": composite_raw_string,
    "extracted_string": composite_extracted_string,
    "list": composite_list,
}

# The agent-facing type-tool names (used by integrity/score-consistency checks).
TYPE_TOOLS = {"evaluate_raw_string", "evaluate_extracted_string", "evaluate_list"}

# Legacy name → function map for the seven underlying metrics. No longer exposed
# as MCP tools, but kept for reference and any non-agentic callers.
METRIC_REGISTRY = {
    "word_iou": word_iou,
    "null_accuracy": null_accuracy,
    "levenshtein": levenshtein_similarity,
    "char_f1": char_f1,
    "set_f1": set_f1,
    "sequence_lcs": sequence_lcs,
    "set_inclusion": set_inclusion,
}

# ---------------------------------------------------------------------------
#  MongoDB connection (lifted from scripts/6b_mongo_eval.py)
# ---------------------------------------------------------------------------
load_dotenv(BASE_DIR / ".env")

_username = os.getenv("MONGO_DB_USERNAME")
_password = os.getenv("MONGO_DB_PASSWORD")

_hosts = [
    "darc-data-shard-00-00.9fjam.mongodb.net:27017",
    "darc-data-shard-00-01.9fjam.mongodb.net:27017",
    "darc-data-shard-00-02.9fjam.mongodb.net:27017",
]
_set_name = "DARC-Data-shard-0"
_uri = (
    f"mongodb://{_username}:{_password}@{','.join(_hosts)}/"
    f"?tls=true&replicaSet={_set_name}&authSource=admin&retryWrites=true&w=majority&appName=DARC-Data"
)

DB_NAME = "usf-internship"
LLM_OUTPUTS_COLL = "llm_outputs"
BENCHMARKS_COLL = "benchmarks"
GROUND_TRUTHS_COLL = "ground_truths"
AGENTIC_EVAL_COLL = "agentic_evaluations"

# Only benchmarks 4–12 carry a `ground_truth` block, so only those are evaluable.
EVALUABLE_BENCHMARK_IDS = [str(i) for i in range(4, 13)]

_client: MongoClient | None = None
_indexes_ready = False


def get_db():
    """Return the project database, reusing a single client across calls."""
    global _client
    if _client is None:
        _client = MongoClient(_uri, server_api=ServerApi("1"))
    return _client[DB_NAME]


def _ensure_indexes(db) -> None:
    """
    Idempotent unique index on the agentic output collection.

    create_index is PyMongo's built-in Collection method (same one used in
    scripts/6b_mongo_eval.py); calling it when the index already exists is a
    no-op. run_id is part of the key on purpose: the same task is run multiple
    times to measure LLM consistency, so each run must produce its own row.
    Without run_id, run 2 would overwrite run 1 and the signal would be lost.
    """
    global _indexes_ready
    if _indexes_ready:
        return
    db[AGENTIC_EVAL_COLL].create_index(
        [("task_id", 1), ("benchmark_id", 1), ("model_id", 1), ("run_id", 1)],
        unique=True,
    )
    _indexes_ready = True


# ---------------------------------------------------------------------------
#  Read tools
# ---------------------------------------------------------------------------

def list_pending_outputs(benchmark_id: str | None = None, limit: int = 20) -> dict:
    """
    Return candidate (task_id, run_id) pairs to evaluate (status == 'processed',
    evaluable benchmark). This is the discovery tool the agent/notebook iterates
    over. run_id is included because one task_id has multiple runs.
    """
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        return {"error": "limit must be an integer", "count": 0, "outputs": []}
    if limit < 1:
        return {"error": "limit must be >= 1", "count": 0, "outputs": []}

    if benchmark_id is not None:
        bid = str(benchmark_id)
        if bid not in EVALUABLE_BENCHMARK_IDS:
            return {
                "error": f"benchmark_id {bid} is not evaluable (no ground_truth). "
                         f"Evaluable: {EVALUABLE_BENCHMARK_IDS}",
                "count": 0,
                "outputs": [],
            }
        bench_filter: Any = bid
    else:
        bench_filter = {"$in": EVALUABLE_BENCHMARK_IDS}

    db = get_db()
    query = {"benchmark_id": bench_filter, "status": "processed"}
    cursor = db[LLM_OUTPUTS_COLL].find(
        query,
        {"task_id": 1, "run_id": 1, "benchmark_id": 1, "model_id": 1,
         "model_name": 1, "image_id": 1},
    ).limit(limit)

    outputs = [
        {
            "task_id": doc.get("task_id"),
            "run_id": doc.get("run_id"),
            "benchmark_id": doc.get("benchmark_id"),
            "model_id": doc.get("model_id"),
            "model_name": doc.get("model_name"),
            "image_id": doc.get("image_id"),
        }
        for doc in cursor
    ]
    return {"count": len(outputs), "outputs": outputs}


def fetch_evaluable_output(task_id: str, run_id: Any = None) -> dict:
    """
    Load one llm_outputs doc and return only what the agent needs to *judge*:
    for each field, the predicted value and the expected (ground-truth) value.

    A task_id can have multiple runs; pass run_id to target a specific one.
    If run_id is omitted, the first matching run is returned.

    The benchmark spec's declared `type` and `metrics` are intentionally NOT
    returned — the agent must infer the appropriate metric from the data itself.
    """
    if not isinstance(task_id, str) or not task_id.strip():
        return {"error": "task_id must be a non-empty string", "fields": []}
    task_id = task_id.strip()

    db = get_db()
    query: dict = {"task_id": task_id}
    if run_id is not None:
        query["run_id"] = run_id
    doc = db[LLM_OUTPUTS_COLL].find_one(query)
    if doc is None:
        return {"error": f"no llm_outputs doc matching {query}", "fields": []}

    benchmark_id = str(doc.get("benchmark_id"))
    image_id = doc.get("image_id")

    benchmark_spec = db[BENCHMARKS_COLL].find_one({"_id": benchmark_id})
    if benchmark_spec is None:
        return {"error": f"benchmark {benchmark_id} not found", "fields": []}

    gt_spec = benchmark_spec.get("ground_truth")
    if not gt_spec:
        return {
            "error": f"benchmark {benchmark_id} has no ground_truth block "
                     f"(not an evaluable benchmark)",
            "fields": [],
        }

    gt_doc = db[GROUND_TRUTHS_COLL].find_one({"_id": image_id})
    if gt_doc is None:
        return {"error": f"ground truth for image {image_id} not found", "fields": []}

    # Output is either a dict (multi-field/structured benchmark) or a bare
    # value (single-field benchmark). Field values are always str or list.
    raw_output = doc.get("output")
    output_is_dict = isinstance(raw_output, dict)
    single_field = len(gt_spec) == 1

    fields = []
    for field_key, field_def in gt_spec.items():
        if output_is_dict:
            predicted = raw_output.get(field_def["output_field"])
        elif single_field:
            predicted = raw_output  # bare value belongs to the lone field
        else:
            predicted = None  # multi-field benchmark but output isn't structured
        expected = _resolve_gt_value(gt_doc, field_def["gt_field"])
        fields.append(
            {
                "field": field_key,
                "predicted": predicted,
                "expected": expected,
            }
        )

    return {
        "task_id": doc.get("task_id"),
        "run_id": doc.get("run_id"),
        "benchmark_id": benchmark_id,
        "benchmark_name": benchmark_spec.get("task_name"),
        "model_id": doc.get("model_id"),
        "model_name": doc.get("model_name"),
        "image_id": image_id,
        "fields": fields,
    }


# ---------------------------------------------------------------------------
#  Write tool
# ---------------------------------------------------------------------------

def save_agentic_evaluation(
    task_id: str,
    benchmark_id: str,
    model_id: str,
    run_id: Any,
    image_id: Any,
    field_evaluations: list,
) -> dict:
    """
    Upsert one agentic evaluation into the agentic_evaluations collection.

    Idempotent on (task_id, benchmark_id, model_id, run_id) so re-evaluating a
    run updates in place, while different runs of the same task each get their
    own row (needed for the LLM-consistency comparison).

    `field_evaluations` is the agent's record of the type-tool it routed to, e.g.
        [{"field": "first_program", "field_type": "raw_string",
          "metric": "evaluate_raw_string",
          "scores": {"composite_score": 0.67, "word_iou": 0.67},
          "rationale": "free-form OCR text"}]
    """
    if not isinstance(task_id, str) or not task_id.strip():
        return {"error": "task_id must be a non-empty string", "saved": False}
    if not isinstance(field_evaluations, list) or not field_evaluations:
        return {"error": "field_evaluations must be a non-empty list", "saved": False}

    db = get_db()
    _ensure_indexes(db)

    filter_key = {
        "task_id": task_id,
        "benchmark_id": str(benchmark_id),
        "model_id": model_id,
        "run_id": run_id,
    }
    eval_doc = {
        **filter_key,
        "image_id": image_id,
        "field_evaluations": field_evaluations,
        "source": "agentic",
        "evaluated_at": datetime.now(timezone.utc),
    }
    db[AGENTIC_EVAL_COLL].replace_one(filter_key, eval_doc, upsert=True)

    return {
        "saved": True,
        "collection": AGENTIC_EVAL_COLL,
        "key": filter_key,
        "n_fields": len(field_evaluations),
    }
