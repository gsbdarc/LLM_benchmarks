"""
tools.py — Business-logic layer for the agentic metric-evaluation MCP server.

Mirrors the role redivis_tools.py plays for the Redivis server: it owns MongoDB
access, input validation, and the metric calculators. The FastMCP server
(server.py) is a thin wrapper that exposes these as tools.

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
from datetime import datetime, timedelta, timezone
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
AGENTIC_RUNS_COLL = "agentic_runs"
# Date-fix task output. Written to, never read by the eval pipeline; the source
# collections (llm_outputs / unified_evaluations) are never modified.
AGENTIC_CORRECTIONS_COLL = "agentic_corrections"

# Identity of one run row = one output (task/run/benchmark/model_under_test) judged
# by one judge config. agent_model_key/prompt_key (ints, not the display strings) are
# used so two variants that share a model/prompt name but differ in kwargs stay distinct.
RUN_KEY_FIELDS = [
    "task_id", "run_id", "benchmark_id", "model_id",
    "backend", "agent_model_key", "prompt_key",
]

# Only benchmarks 4–12 carry a `ground_truth` block, so only those are evaluable.
EVALUABLE_BENCHMARK_IDS = [str(i) for i in range(4, 13)]

_client: MongoClient | None = None
_indexes_ready = False


def get_db() -> Any:
    """Return the project database, reusing a single client across calls."""
    global _client
    if _client is None:
        _client = MongoClient(_uri, server_api=ServerApi("1"))
    return _client[DB_NAME]


def _ensure_indexes(db: Any) -> None:
    """
    Idempotent indexes on the agentic_evaluations collection.

    The unique key is `eval_id` (the mapping's per-(output × judge) surrogate), via a
    PARTIAL index so it applies only to real eval_ids — letting the same output graded
    by different judges keep one doc each. The LEGACY unique index on
    (task_id, benchmark_id, model_id, run_id) forced one doc per output and collapsed
    multi-judge verdicts, so it is dropped if present; a plain (non-unique) index on
    those four is kept for query/join performance. create_index is idempotent.
    """
    global _indexes_ready
    if _indexes_ready:
        return
    coll = db[AGENTIC_EVAL_COLL]
    # Drop superseded UNIQUE indexes: the 4-field legacy (collapsed multi-judge), and the
    # eval_id-only one (predates versioning — it would forbid the same eval under a new
    # code_version). The current key is (eval_id, git_commit).
    existing = {ix["name"]: ix for ix in coll.list_indexes()}
    for name in ("task_id_1_benchmark_id_1_model_id_1_run_id_1", "eval_id_1"):
        if name in existing and existing[name].get("unique"):
            coll.drop_index(name)
    coll.create_index(
        [("eval_id", 1), ("git_commit", 1)], unique=True,
        partialFilterExpression={"eval_id": {"$type": ["int", "long"]}},
    )
    coll.create_index([("task_id", 1), ("benchmark_id", 1), ("model_id", 1), ("run_id", 1)])
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
        # An absent value for a NON-list field is stored as an empty list []; that reads
        # as an empty *list* and mis-routes judges to evaluate_list. Represent absence as
        # "" for non-list fields so it reads as an absent extracted/raw value. (List fields
        # keep [] — a genuine empty collection.) `type` is the benchmark's own declared
        # field type; it is still not surfaced to the agent, only used here to clean the value.
        if expected == [] and field_def.get("type") != "list":
            expected = ""
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
#  Date-fix task: read the model's own three answers, derive the guide date,
#  record a correction. A separate task from metric evaluation above — it repairs
#  benchmark-3 (tv_guide_date) outputs, which are DERIVED from benchmarks 1 and 2
#  (newspaper_date + day_of_week) rather than read straight off the page.
# ---------------------------------------------------------------------------

# benchmark_id -> the field this task calls it. Fixed by the task, not configurable.
DATE_FIX_BENCHMARKS = {"1": "newspaper_date", "2": "day_of_week", "3": "tv_guide_date"}

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
_WEEKDAY_INDEX = {d.lower(): i for i, d in enumerate(WEEKDAYS)}

# Formats seen across ground_truths and llm_outputs (e.g. "Dec 17 2000",
# "December 20 2000", "2014-08-02", "12/17/2000").
_DATE_FORMATS = ("%b %d %Y", "%B %d %Y", "%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y",
                 "%d %b %Y", "%d %B %Y")

# The corpus spans 1994-2016 newspapers. A publication year outside this window means
# the extracted value is corrupt, not that the guide is unusual — see compute_guide_date.
PLAUSIBLE_YEARS = (1950, 2035)

# Models often prefix the weekday ("Sun, Feb 22, 2015"); it carries no extra
# information once we have the date, so it is dropped before parsing.
_WEEKDAY_PREFIXES = frozenset(
    [d.lower() for d in WEEKDAYS] + [d.lower()[:3] for d in WEEKDAYS]
)


def parse_date(value: Any) -> datetime | None:
    """Parse a date string in any format this corpus uses, else None.

    Shared by the tool, the work-list builder and the scorer, so "is this the right
    date" is judged the same way everywhere. Tolerates the shapes models actually
    emit — "Sun, Feb 22, 2015", "Monday, Oct. 15, 2007", "2014-08-02" — by dropping a
    leading weekday and stripping punctuation before matching.

    Deliberately returns None (not a guess) for values that name no single date:
      * no year at all ("Monday, Oct. 15") — resolving it would mean inventing one
      * a range ("Oct. 30-Nov. 5, 1994") — several dates, not one
    Those are genuinely unusable answers, and the caller decides what that means.
    """
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str) or not value.strip():
        return None
    s = value.strip().replace(",", " ").replace(".", " ")
    s = " ".join(s.split())                       # collapse the gaps punctuation left
    head, _, rest = s.partition(" ")
    if rest and head.lower().rstrip() in _WEEKDAY_PREFIXES:
        s = rest                                  # "Sun Feb 22 2015" -> "Feb 22 2015"
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def compute_guide_date(publication_date: Any, day_of_week: Any) -> dict:
    """
    Derive the date a TV guide covers from the newspaper's publication date and
    the day of the week the guide is for.

    Rule: the guide date is the FIRST occurrence of `day_of_week` falling ON OR
    AFTER `publication_date`. Returns {"date": "YYYY-MM-DD", ...} on success.

    Returns {"ambiguous": <reason>} — and NO date — when the inputs don't describe
    a single day: a weekday RANGE ("Monday-Friday") covers a span of dates, and an
    unparseable date or weekday can't be resolved at all. An ambiguous result means
    the case needs a human; record it as an abstention rather than guessing.
    """
    pub = parse_date(publication_date)
    if pub is None:
        return {"ambiguous": f"could not parse publication_date {publication_date!r}"}
    # The prerequisite can itself be corrupt: one model reported the publication date of
    # a 2002 paper as "1902-05-26". Deriving from that produced a confidently wrong date
    # that overwrote a CORRECT answer. This corpus is 1994-2016, so a year far outside it
    # means the input is untrustworthy — abstain and let a human look rather than compute.
    if not PLAUSIBLE_YEARS[0] <= pub.year <= PLAUSIBLE_YEARS[1]:
        return {"ambiguous": f"publication_date {publication_date!r} has implausible year "
                             f"{pub.year} (expected {PLAUSIBLE_YEARS[0]}-{PLAUSIBLE_YEARS[1]}); "
                             "the source value looks corrupt"}

    if not isinstance(day_of_week, str) or not day_of_week.strip():
        return {"ambiguous": f"day_of_week {day_of_week!r} is empty"}
    dow = day_of_week.strip()
    named = [d for d in WEEKDAYS if d.lower() in dow.lower()]
    if len(named) > 1 or any(sep in dow.lower() for sep in ("-", " to ", "through", "thru", "&")):
        return {"ambiguous": f"day_of_week {dow!r} spans multiple days, not one date"}
    if dow.lower() not in _WEEKDAY_INDEX:
        if not named:
            return {"ambiguous": f"day_of_week {dow!r} is not a weekday name"}
        dow = named[0]

    offset = (_WEEKDAY_INDEX[dow.lower()] - pub.weekday()) % 7
    guide = pub + timedelta(days=offset)
    result = {
        "date": guide.strftime("%Y-%m-%d"),
        "publication_weekday": WEEKDAYS[pub.weekday()],
        "days_after_publication": offset,
        "rule": "first occurrence of the weekday on or after the publication date",
    }
    # Offset 0 means the paper was published ON the guide's own weekday, and the corpus
    # genuinely disagrees about what that means: four images have the guide covering
    # publication day itself, two have it covering that weekday a week later. The rule
    # picks same-day, which is right more often — but it IS a guess, so say so instead of
    # returning it with the same confidence as an unambiguous case.
    if offset == 0:
        result["needs_review"] = True
        result["review_reason"] = (
            f"the paper was published on a {dow}, the same weekday the guide covers. "
            "Comparable cases are split between the guide covering publication day itself "
            "and the same weekday a week later, so this date is the more likely of two "
            "readings rather than a certainty — flag it for a human."
        )
    return result


def get_guide_date_case(task_id: str, run_id: Any = None) -> dict:
    """
    Load one TV-guide-date case: what THIS model answered for THIS image, across
    the three related benchmarks.

    Given the task_id of a tv_guide_date output, returns that answer alongside the
    same model's own `newspaper_date` and `day_of_week` answers for the same image
    and run — the two values the guide date is derived from.

    Ground truth is deliberately NOT returned. Judge whether `tv_guide_date` is
    consistent with the other two values; do not try to recall what the real date was.
    """
    if not isinstance(task_id, str) or not task_id.strip():
        return {"error": "task_id must be a non-empty string"}
    task_id = task_id.strip()

    db = get_db()
    query: dict = {"task_id": task_id}
    if run_id is not None:
        query["run_id"] = run_id
    anchor = db[LLM_OUTPUTS_COLL].find_one(query)
    if anchor is None:
        return {"error": f"no llm_outputs doc matching {query}"}

    # The three benchmarks are separate outputs by the same model for the same image/run.
    siblings = db[LLM_OUTPUTS_COLL].find({
        "model_id": anchor.get("model_id"),
        "image_id": anchor.get("image_id"),
        "run_id": anchor.get("run_id"),
        "benchmark_id": {"$in": list(DATE_FIX_BENCHMARKS)},
    })
    answers = {DATE_FIX_BENCHMARKS[d["benchmark_id"]]: d.get("output") for d in siblings
               if d.get("benchmark_id") in DATE_FIX_BENCHMARKS}

    missing = [f for f in DATE_FIX_BENCHMARKS.values() if f not in answers]
    case = {
        "task_id": task_id,
        "run_id": anchor.get("run_id"),
        "image_id": anchor.get("image_id"),
        "model_id": anchor.get("model_id"),
        "model_name": anchor.get("model_name"),
        "newspaper_date": answers.get("newspaper_date"),
        "day_of_week": answers.get("day_of_week"),
        "tv_guide_date": answers.get("tv_guide_date"),
        **({"missing_fields": missing} if missing else {}),
    }

    # Self-evident corruption is flagged HERE, without consulting ground truth: one
    # model reported a 2002 paper's date as "1902-05-26", and deriving from that
    # overwrote a correct answer. An 1902 newspaper is impossible on its face, so the
    # case is served with a warning telling the agent to abstain rather than compute.
    # (Whether an input merely DISAGREES with the answer key is a selection question —
    # see registry/create_date_fix_mapping.py — because checking it here would tell the
    # agent its inputs are verified, which it could never know in production.)
    published = parse_date(case["newspaper_date"])
    if published is not None and not PLAUSIBLE_YEARS[0] <= published.year <= PLAUSIBLE_YEARS[1]:
        case["input_warning"] = (
            f"newspaper_date {case['newspaper_date']!r} has an implausible year "
            f"({published.year}); this corpus is {PLAUSIBLE_YEARS[0]}-{PLAUSIBLE_YEARS[1]}. "
            "The source value looks corrupt, so no date can be derived from it — record "
            "this case with action=\"abstained\" for a human to check."
        )
    return case


def save_correction(
    task_id: str,
    action: str,
    original_value: Any,
    final_value: Any,
    reason: str,
    image_id: Any = None,
    run_id: Any = None,
    model_id: Any = None,
    computed_date: Any = None,
    needs_review: bool = False,
    review_reason: str | None = None,
    eval_id: int | None = None,
    git_commit: str | None = None,
) -> dict:
    """
    Record the outcome for one reviewed case in the agentic_corrections collection.

    Call this EXACTLY ONCE per case, whatever you decided — the collection is meant
    to be usable on its own, so a reviewed case with no row is a hole in the data.

    `action` must be one of:
      - "corrected"  — tv_guide_date was inconsistent; final_value is the fixed date
      - "confirmed"  — tv_guide_date was already right; final_value repeats it
      - "abstained"  — it can't be determined (e.g. compute_guide_date returned
                       ambiguous); final_value keeps the original and `reason` says why

    `final_value` is the value a downstream consumer should use. `computed_date` is
    what compute_guide_date returned, kept as the basis for the decision. Nothing in
    llm_outputs or unified_evaluations is ever modified.

    Set `needs_review=True` with a `review_reason` whenever your answer is a judgment
    call rather than a certainty — in particular when compute_guide_date returned
    "needs_review", or when the values were odd enough that a person should confirm.
    This is separate from `action`: a corrected OR confirmed row can still want review.
    It costs nothing to flag, and an unflagged wrong answer is far more expensive than a
    flagged uncertain one.
    """
    valid = ("corrected", "confirmed", "abstained")
    if action not in valid:
        return {"error": f"action must be one of {valid}, got {action!r}", "saved": False}
    if not isinstance(task_id, str) or not task_id.strip():
        return {"error": "task_id must be a non-empty string", "saved": False}
    if not isinstance(reason, str) or not reason.strip():
        return {"error": "reason must be a non-empty string", "saved": False}

    db = get_db()
    filter_key = (
        {"eval_id": eval_id, "git_commit": git_commit}
        if eval_id is not None
        else {"task_id": task_id.strip(), "run_id": run_id, "git_commit": git_commit}
    )
    doc = {
        "eval_id": eval_id,
        "git_commit": git_commit,
        "task_id": task_id.strip(),
        "run_id": run_id,
        "image_id": image_id,
        "model_id": model_id,
        "action": action,
        "original_value": original_value,
        "final_value": final_value,
        "computed_date": computed_date,
        "needs_review": bool(needs_review),
        "review_reason": (review_reason or "").strip() or None,
        "reason": reason.strip(),
        "source": "agentic",
        "evaluated_at": datetime.now(timezone.utc),
    }
    db[AGENTIC_CORRECTIONS_COLL].replace_one(filter_key, doc, upsert=True)
    return {"saved": True, "collection": AGENTIC_CORRECTIONS_COLL,
            "key": filter_key, "action": action}


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
    eval_id: int | None = None,
    git_commit: str | None = None,
) -> dict:
    """
    Upsert one agentic evaluation into the agentic_evaluations collection.

    Keyed on (`eval_id`, `git_commit`) when eval_id is present — the mapping's surrogate
    id (unique per output × judge config) plus the code_version, so the SAME eval under a
    new code version coexists rather than overwriting the old result, and different judges
    of one output each keep a doc. eval_id is the shared join key with agentic_runs. Without
    eval_id (discovery/dev) it falls back to (task_id, benchmark_id, model_id, run_id,
    git_commit). `git_commit` is stamped client-side (the agent loop), like eval_id.

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

    filter_key = (
        {"eval_id": eval_id, "git_commit": git_commit}
        if eval_id is not None
        else {"task_id": task_id, "benchmark_id": str(benchmark_id),
              "model_id": model_id, "run_id": run_id, "git_commit": git_commit}
    )
    eval_doc = {
        "eval_id": eval_id,
        "git_commit": git_commit,
        "task_id": task_id,
        "benchmark_id": str(benchmark_id),
        "model_id": model_id,
        "run_id": run_id,
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


def save_run_row(row: dict[str, Any]) -> dict[str, Any]:
    """Mirror one flat run-observability row into the agentic_runs collection.

    This is the central, team-queryable copy of what the local Parquet sink writes
    (perf, tokens, save_success, selection_accuracy, weave_trace_url, ...). Keyed on
    (`eval_id`, `git_commit`) when eval_id is present — the shared join key with
    agentic_evaluations plus the code_version — else the RUN_KEY_FIELDS composite +
    git_commit. So a re-run under a NEW code version coexists (history preserved), while
    re-running the SAME version updates in place. `stored_at` records the mirror time.
    """
    db = get_db()
    filter_key = _run_filter_key(row)
    doc = {**row, "stored_at": datetime.now(timezone.utc)}
    db[AGENTIC_RUNS_COLL].replace_one(filter_key, doc, upsert=True)
    return {"saved": True, "collection": AGENTIC_RUNS_COLL, "key": filter_key}


def _run_filter_key(row: dict[str, Any]) -> dict[str, Any]:
    """The version-aware uniqueness key for an agentic_runs doc: (eval_id, git_commit)
    when eval_id is present, else the RUN_KEY_FIELDS composite + git_commit."""
    eval_id = row.get("eval_id")
    gc = row.get("git_commit")
    if eval_id is not None:
        return {"eval_id": eval_id, "git_commit": gc}
    return {**{k: row.get(k) for k in RUN_KEY_FIELDS}, "git_commit": gc}


def run_exists(row: dict[str, Any]) -> bool:
    """True if agentic_runs already has this run for its (identity + code_version).

    Lets a batch SKIP re-running a (task × judge × prompt) under a code_version that was
    already evaluated — so re-runs don't re-spend API or clobber, and only genuinely new
    (task × version) work executes. Needs the same fields as _run_filter_key on `row`.

    Only a NON-errored run counts as "exists": a previously errored run (transient API
    failure) is re-runnable, so it does not block its own retry.
    """
    query = {**_run_filter_key(row), "stopped_reason": {"$ne": "error"}}
    return get_db()[AGENTIC_RUNS_COLL].count_documents(query, limit=1) > 0
