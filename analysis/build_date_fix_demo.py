"""
build_date_fix_demo.py — the date-fix demo page.

Sibling of build_dashboard.py, for the OTHER task. Where that one reports metric-eval
routing across many judges, this reports date-repair batches to a non-specialist
audience: what each judge fixed, what it left alone, what it handed back to a human,
what it cost, and the step-by-step trace behind every row.

Self-contained HTML with the data embedded as JSON — no server, no network, no libraries,
so it opens from disk in a meeting room.

    python -m analysis.build_date_fix_demo                    # -> images/date_fix_demo.html
    python -m analysis.build_date_fix_demo --open

Every judge that ran the prompt is embedded, and the page has one judge switcher scoping
the per-row sections; the summary and the tool paths always show all judges side by side.
Ground truth lives in the work-list CSVs (the agent never sees it), so it can only be
joined here.
"""

from __future__ import annotations

import argparse
import json
import webbrowser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = Path(__file__).resolve().parent / "date_fix_demo_template.html"
DEFAULT_OUT = REPO_ROOT / "images" / "date_fix_demo.html"
MAPPING_GLOB = "date_fix_mapping*.csv"
PLACEHOLDER = "__DATEFIX_DATA__"
PROMPT_NAME = "date_fix_v1"

# Reader-facing labels. Stack order is the validated palette order (blue, aqua, violet, red).
OUTCOMES = [
    {"key": "fixed_correct", "label": "Fixed", "icon": "✔",
     "blurb": "was wrong; the agent corrected it to the right date"},
    {"key": "confirmed_correct", "label": "Confirmed", "icon": "=",
     "blurb": "was already right; the agent left it alone"},
    {"key": "flagged", "label": "Left for a person", "icon": "⚑",
     "blurb": "no single date is correct, so the agent declined to invent one"},
    {"key": "wrong", "label": "Wrong", "icon": "✖",
     "blurb": "the agent's answer does not match the truth"},
]
GROUPS = [
    {"key": "wrong", "label": "Needed fixing",
     "description": "The date was wrong, and the two values it derives from were read correctly."},
    {"key": "control", "label": "Already correct",
     "description": "The date was already right. Any change here would be damage."},
    {"key": "range", "label": "No single answer",
     "description": "The guide covers a span of days, so no one date is correct."},
]


def _display_outcome(row: dict[str, Any]) -> str:
    """Collapse the scorer's six outcomes into the four a reader needs.

    Being FLAGGED for review is deliberately NOT folded in here. It is orthogonal to
    being right — most flagged rows are correct — and folding them together would report
    them as unresolved and hide the work the agent did. Flagging is carried separately.
    """
    outcome = row.get("fix_outcome")
    if outcome in ("fixed_wrong", "confirmed_wrong", "no_action"):
        return "wrong"
    if outcome == "abstained":
        return "flagged"
    return outcome or "wrong"


def _bucket(row: dict[str, str], parse_date: Any) -> str:
    """Which group a work-list row belongs to (recomputed, not stored)."""
    expected = parse_date(row.get("expected_date"))
    if expected is None:
        return "range"
    return "control" if parse_date(row.get("original_value")) == expected else "wrong"


def _flow(paths: list[list[str]]) -> dict[str, Any]:
    """Nodes per call position and weighted transitions, for the path diagram."""
    columns: list[dict[str, int]] = []
    edges: dict[tuple[int, str, str], int] = {}
    for path in paths:
        for i, tool in enumerate(path):
            while len(columns) <= i:
                columns.append({})
            columns[i][tool] = columns[i].get(tool, 0) + 1
            if i:
                key = (i - 1, path[i - 1], tool)
                edges[key] = edges.get(key, 0) + 1
    return {
        "columns": [[{"tool": t, "count": c} for t, c in sorted(col.items(), key=lambda kv: -kv[1])]
                    for col in columns],
        "edges": [{"col": k[0], "from": k[1], "to": k[2], "count": v} for k, v in edges.items()],
    }


def build_snapshot(judge: str | None = None) -> dict[str, Any]:
    """Join runs, corrections and the work lists into the page payload."""
    from agent_eval.prompts import resolve_prompt
    from agent_eval.registry import mapping as mapping_mod
    from agent_eval.registry.create_date_fix_mapping import build_candidates
    from agent_eval.tools import get_db, parse_date

    name, system, user, _ = resolve_prompt(PROMPT_NAME)
    db = get_db()

    # Each judge has its own work list (eval_ids are unique per judge x output).
    work: dict[int, dict[str, Any]] = {}
    for path in sorted((REPO_ROOT / "inputs").glob(MAPPING_GLOB)):
        work.update({int(r["eval_id"]): r for r in mapping_mod.read_csv(path)})

    all_runs = list(db["agentic_runs"].find({"prompt_name": PROMPT_NAME}))
    if not all_runs:
        raise SystemExit(f"no {PROMPT_NAME} runs in agentic_runs — run the batch first")
    corrections = {d.get("eval_id"): d for d in db["agentic_corrections"].find()}

    models = json.loads((REPO_ROOT / "inputs" / "models.json").read_text())
    model_names = {k: v.get("model") for k, v in models.items()}
    images = {}
    for doc in db["llm_outputs"].find({"benchmark_id": "3"},
                                      {"task_id": 1, "run_id": 1, "image_id": 1}):
        images[(doc["task_id"], doc["run_id"])] = doc.get("image_id")

    by_judge: dict[str, list[dict[str, Any]]] = {}
    for run in all_runs:
        by_judge.setdefault(run.get("model") or "?", []).append(run)

    judges = []
    for model, runs in by_judge.items():
        rows, per_group = [], {g["key"]: {"n": 0, **{o["key"]: 0 for o in OUTCOMES}} for g in GROUPS}
        for run in sorted(runs, key=lambda r: r.get("eval_id") or 0):
            eval_id = run.get("eval_id")
            job = work.get(eval_id, {})
            saved = corrections.get(eval_id, {})
            group = _bucket(job, parse_date) if job else "wrong"
            shown = _display_outcome(run)
            per_group[group]["n"] += 1
            per_group[group][shown] += 1
            try:
                trace = json.loads(run.get("steps_json") or "[]")
            except (TypeError, json.JSONDecodeError):
                trace = []
            try:
                path = json.loads(run.get("tool_sequence_json") or "[]")
            except (TypeError, json.JSONDecodeError):
                path = []
            rows.append({
                "eval_id": eval_id,
                "image_id": images.get((job.get("task_id"), int(job.get("run_id") or 0))),
                "model_under_test": model_names.get(str(job.get("model_id")), job.get("model_id")),
                "group": group,
                "display_outcome": shown,
                "fix_outcome": run.get("fix_outcome"),
                "action": saved.get("action"),
                "needs_review": bool(run.get("fix_needs_review")),
                # The distinction that matters: a wrong answer the agent did NOT flag is a
                # silent error; a wrong answer it flagged reached a human anyway.
                "confidently_wrong": shown == "wrong" and not run.get("fix_needs_review"),
                "review_reason": saved.get("review_reason"),
                "reason": saved.get("reason"),
                "original_value": job.get("original_value"),
                "final_value": saved.get("final_value"),
                "expected_date": job.get("expected_date"),
                "cost": run.get("total_dollar_cost"),
                "wall_time": run.get("wall_time_total"),
                "steps": run.get("steps"),
                "total_tokens": run.get("total_tokens"),
                "reasoning_tokens": run.get("reasoning_tokens"),
                "path": path,
                "trace": trace,
            })

        n = len(rows)
        cost = sum(r["cost"] or 0 for r in rows)
        waits = sorted(r["wall_time"] or 0 for r in rows)
        def pct(p: float) -> float:
            return round(waits[min(int(p * len(waits)), len(waits) - 1)], 1) if waits else 0.0
        tokens = sum(r["total_tokens"] or 0 for r in rows)
        judges.append({
            "model": model, "n": n,
            "fixed": per_group_total(per_group, "fixed_correct"),
            "confirmed": per_group_total(per_group, "confirmed_correct"),
            "abstained": per_group_total(per_group, "flagged"),
            "wrong": per_group_total(per_group, "wrong"),
            "regressions": sum(1 for r in runs if r.get("fix_regression")),
            "flagged": sum(1 for r in rows if r["needs_review"]),
            "flagged_but_correct": sum(1 for r in rows if r["needs_review"]
                                       and r["display_outcome"] in ("fixed_correct", "confirmed_correct")),
            "wrong_and_flagged": sum(1 for r in rows if r["display_outcome"] == "wrong" and r["needs_review"]),
            "confidently_wrong": sum(1 for r in rows if r["confidently_wrong"]),
            "wrong_before": sum(1 for r in rows if r["group"] == "wrong"),
            "controls": sum(1 for r in rows if r["group"] == "control"),
            "cost": round(cost, 4), "per_answer": round(cost / n, 5) if n else 0,
            "total_dollar_cost": round(cost, 4),
            "median_s": pct(0.5), "p90_s": pct(0.9),
            "min_s": round(waits[0], 1) if waits else 0, "max_s": round(waits[-1], 1) if waits else 0,
            "productive_s": round(sum(r.get("llm_time_productive") or 0 for r in runs), 1),
            "retry_s": round(sum(r.get("llm_retry_wait") or 0 for r in runs), 1),
            "attempts": sum(r.get("llm_attempts") or 0 for r in runs),
            "steps": sum(r["steps"] or 0 for r in rows),
            "tokens": tokens, "tokens_per_answer": round(tokens / n) if n else 0,
            "reasoning_tokens": sum(r["reasoning_tokens"] or 0 for r in rows),
            "traces": sum(1 for r in rows if r["trace"]),
            "n_paths": len({tuple(r["path"]) for r in rows}),
            "groups": [{**g, **per_group[g["key"]]} for g in GROUPS],
            "flow": _flow([r["path"] for r in rows]),
            "paths": [{"path": list(p), "count": c} for p, c in
                      sorted(count_paths(rows).items(), key=lambda kv: -kv[1])],
            "wall_times": [round(r["wall_time"] or 0, 1) for r in rows],
            "rows": rows,
        })
    judges.sort(key=lambda j: j["per_answer"])

    # What it cost to PRODUCE these answers originally, averaged over the whole benchmark
    # (all rows priced) rather than the sample — a partial join would skew the comparison.
    agg = list(db["unified_evaluations"].aggregate([
        {"$match": {"benchmark_id": "3"}},
        {"$group": {"_id": None, "n": {"$sum": 1}, "cost": {"$sum": "$total_dollar_cost"}}},
    ]))
    orig = agg[0] if agg else {"n": 0, "cost": 0}
    orig_per = (orig["cost"] / orig["n"]) if orig["n"] else 0
    candidates, scope = build_candidates()

    focus = judge or judges[0]["model"]
    if focus not in {j["model"] for j in judges}:
        raise SystemExit(f"no runs for judge {focus!r}; have {[j['model'] for j in judges]}")

    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "git_commit": all_runs[0].get("git_commit"),
        "prompt": {"name": name, "system": system, "user": user},
        "outcomes": OUTCOMES,
        "group_meta": GROUPS,
        "judges": judges,
        "default_judge": focus,
        "original": {
            "n": orig["n"], "total": round(orig["cost"], 2), "per_answer": round(orig_per, 5),
            "flagged_free": scope["flagged_by_free_check"],
            "target": len(candidates["wrong"]),
        },
    }


def per_group_total(per_group: dict[str, dict[str, int]], key: str) -> int:
    return sum(g[key] for g in per_group.values())


def count_paths(rows: list[dict[str, Any]]) -> dict[tuple[str, ...], int]:
    out: dict[tuple[str, ...], int] = {}
    for r in rows:
        out[tuple(r["path"])] = out.get(tuple(r["path"]), 0) + 1
    return out


def render_html(snapshot: dict[str, Any], template_path: str | Path = TEMPLATE) -> str:
    """Inject the payload, escaping `<` so an embedded `</script>` can't break out."""
    template = Path(template_path).read_text()
    if PLACEHOLDER not in template:
        raise ValueError(f"template missing placeholder {PLACEHOLDER}")
    return template.replace(PLACEHOLDER, json.dumps(snapshot, default=str).replace("<", "\\u003c"))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="analysis.build_date_fix_demo")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--judge", default=None,
                   help="judge selected on load (default: the cheapest that ran)")
    p.add_argument("--open", action="store_true", help="open the file in a browser")
    args = p.parse_args(argv)

    snapshot = build_snapshot(judge=args.judge)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_html(snapshot))
    print(f"wrote {out}")
    for j in snapshot["judges"]:
        print(f"  {j['model']:<20} {j['n']:>3} answers  fixed {j['fixed']}  wrong {j['wrong']} "
              f"(confidently wrong {j['confidently_wrong']})  flagged {j['flagged']}  "
              f"${j['cost']:.2f}")
    if args.open:
        webbrowser.open(out.resolve().as_uri())


if __name__ == "__main__":
    main()
