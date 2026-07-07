"""
scorers.py — weave.Evaluation scorers (notebook cell a0bad0e1, unchanged logic).

Each scorer takes `output` (the run_agent result dict) plus dataset-row fields and
returns a dict of numbers Weave aggregates across the frozen dataset.
"""

from __future__ import annotations

import csv
from pathlib import Path

from .config import MCP_DIR
from .integrity import check_score_consistency, extract_saved_evaluation
from .observability import op

GOLD_CSV = MCP_DIR / "gold_metrics.csv"


def _load_gold_metrics(path=GOLD_CSV):
    """Load the gold field_type keyed by (benchmark_id, field_name). {} if missing.

    With the composite consolidation, routing accuracy is graded against the
    `field_type` column (raw_string/extracted_string/list) — the one correct data
    shape per field — not the old per-metric `gold_metric`.
    """
    path = Path(path)
    if not path.exists():
        print(f"Warning: {path} not found — run `python3 generate_gold_metrics.py` first.")
        return {}
    gold = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            # Fall back to gold_metric only if an older CSV lacks field_type.
            gold[(row["benchmark_id"], row["field_name"])] = (
                row.get("field_type") or row.get("gold_metric")
            )
    return gold


@op
def save_success_scorer(output):
    _, save_count = extract_saved_evaluation(output.get("messages", []))
    return {
        # >= 1 (not == 1): a row that errored on its first save then retried ends
        # with save_count=2, still a success — only 0 saves is a real failure.
        "save_success": save_count >= 1 and output.get("stopped_reason") == "answered",
        "save_count": save_count,
    }


@op
def score_consistency_scorer(output):
    return check_score_consistency(output.get("messages", []))


@op
def efficiency_scorer(output):
    return {
        "steps": output.get("steps", 0),
        "tool_errors": sum(output.get("tool_errors_by_name", {}).values()),
        "total_tokens": output.get("usage", {}).get("total_tokens", 0),
        # new derived signals also surfaced per-row in the evaluation view
        "tokens_per_sec": output.get("tokens_per_sec"),
        "peak_context": output.get("peak_context"),
    }


def _chosen_field_type(fe):
    """The data shape the agent routed a field to.

    Prefer an explicit "field_type"; otherwise derive it from the type-tool name
    recorded under "metric" (e.g. "evaluate_list" -> "list").
    """
    ft = fe.get("field_type")
    if ft:
        return ft
    metric = fe.get("metric") or ""
    prefix = "evaluate_"
    return metric[len(prefix):] if metric.startswith(prefix) else metric


@op
def selection_accuracy_scorer(output, benchmark_id=None, gold=None, **kwargs):
    """Fraction of fields the agent routed to the correct composite type-tool.

    Graded against the gold `field_type` (raw_string/extracted_string/list).
    Returns None until gold_metrics.csv exists. `gold` may be injected for
    testing; otherwise it is loaded from the CSV.
    """
    if gold is None:
        gold = _load_gold_metrics()
    if not gold or benchmark_id is None:
        return None

    saves, _ = extract_saved_evaluation(output.get("messages", []))
    if not saves:
        return None

    field_evals = saves[0].get("field_evaluations", [])
    scoreable = [fe for fe in field_evals if (str(benchmark_id), fe.get("field")) in gold]
    if not scoreable:
        return None

    correct = sum(
        1 for fe in scoreable
        if _chosen_field_type(fe) == gold[(str(benchmark_id), fe.get("field"))]
    )
    return {
        "selection_accuracy": correct / len(scoreable),
        "correct": correct,
        "total": len(scoreable),
    }
