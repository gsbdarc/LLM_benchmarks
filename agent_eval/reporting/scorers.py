"""
scorers.py — weave.Evaluation scorers (notebook cell a0bad0e1, unchanged logic).

Each scorer takes `output` (the run_agent result dict) plus dataset-row fields and
returns a dict of numbers Weave aggregates across the frozen dataset.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Optional, Union

from ..config import PKG_DIR
from .integrity import check_score_consistency, extract_saved_evaluation, save_outcome
from .observability import op

GOLD_CSV = PKG_DIR / "gold_metrics.csv"


def _load_gold_metrics(path: Union[str, Path] = GOLD_CSV) -> dict[tuple[str, str], Any]:
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
def save_success_scorer(output: dict[str, Any]) -> dict[str, Any]:
    """Score whether the run saved an evaluation and answered (save success/count/failed)."""
    # Shared with integrity.run_integrity_report so the two never drift: success
    # means >=1 SUCCESSFUL save; save_failed preserves the first-attempt-failed nuance.
    attempts, failed, ok = save_outcome(output.get("messages", []))
    return {
        "save_success": ok >= 1 and output.get("stopped_reason") == "answered",
        "save_count": attempts,
        "save_failed": failed,
    }


@op
def score_consistency_scorer(output: dict[str, Any]) -> dict[str, Any]:
    """Score whether every saved score traces back to a prior metric-tool result."""
    return check_score_consistency(output.get("messages", []))


@op
def efficiency_scorer(output: dict[str, Any]) -> dict[str, Any]:
    """Score run efficiency: steps, tool errors, tokens, and derived throughput signals."""
    return {
        "steps": output.get("steps", 0),
        "tool_errors": sum(output.get("tool_errors_by_name", {}).values()),
        "total_tokens": output.get("usage", {}).get("total_tokens", 0),
        # new derived signals also surfaced per-row in the evaluation view
        "tokens_per_sec": output.get("tokens_per_sec"),
        "peak_context": output.get("peak_context"),
    }


def _chosen_field_type(fe: dict[str, Any]) -> str:
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
def selection_accuracy_scorer(
    output: dict[str, Any],
    benchmark_id: Optional[Any] = None,
    gold: Optional[dict[tuple[str, str], Any]] = None,
    **kwargs: Any,
) -> Optional[dict[str, Any]]:
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
