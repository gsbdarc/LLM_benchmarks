"""
build_dashboard.py — generate a self-contained HTML dashboard from the run rows.

Follows the team's "Stanford AI API Token Usage Dashboard" convention: a single
HTML file with the data embedded as JSON, rendered by vanilla JS + inline SVG —
no server, no ports, no external libraries. Open it directly or drop it in a
shared web dir.

    python -m analysis.build_dashboard                 # -> images/agent_dashboard.html
    python -m analysis.build_dashboard --open          # also open in a browser

The snapshot embeds, besides the per-run rows:
  * a prompt registry (agent system+user prompt text, keyed by prompt_name) so a
    run renders the actual prompt it used;
  * a static glossary of agent-facing terms;
  * short LLM-written "why this path" blurbs for the top tool-call paths, built
    from each run's persisted reasoning. Summaries are optional and degrade
    gracefully (skipped) when the summarizer endpoint is unreachable.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import webbrowser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from analysis.queries import connect
from agent_eval import config
from agent_eval.prompts import eval_user_prompt, resolve_prompt

REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = Path(__file__).resolve().parent / "dashboard_template.html"
DEFAULT_OUT = REPO_ROOT / "images" / "agent_dashboard.html"
PLACEHOLDER = "__AGENTVIZ_DATA__"
DEFAULT_CACHE = REPO_ROOT / "outputs" / ".path_summary_cache.json"

# The three composite type-tools (for the routing / glossary views).
TYPE_TOOLS = ["evaluate_raw_string", "evaluate_extracted_string", "evaluate_list"]
METRIC_EVAL_PROMPT_PREFIX = "composite_"

# Columns embedded per run. Kept explicit so the payload is stable and small.
# NOTE: reasoning_json is intentionally NOT embedded per-run (it can be large);
# it is read separately at build time to generate the path summaries.
RUN_COLUMNS = [
    "eval_id", "task_id", "run_id", "benchmark_id", "model_id",
    "backend", "framework", "model", "agent_model_key",
    "temperature", "gpu_type", "reasoning_level", "prompt_name", "prompt_key", "prompt_hash",
    "git_commit",
    "concurrency", "steps", "stopped_reason", "error_detail",
    "n_tool_calls", "n_metric_calls", "n_tool_errors", "tool_sequence_json",
    "prompt_tokens", "completion_tokens", "total_tokens",
    "input_dollar_cost", "output_dollar_cost", "total_dollar_cost",
    "tokens_per_sec", "peak_context", "wall_time_total", "llm_time_total",
    "llm_time_productive", "llm_retry_wait", "llm_attempts", "reasoning_tokens",
    "overhead_time",
    "save_success", "save_count", "save_failed", "score_consistent", "selection_accuracy",
    "routing_path_correct", "routing_path_reason",
    "fix_outcome", "fix_regression", "fix_needs_review",
    "gpu_cache_usage_end", "requests_running_end", "weave_trace_url",
]

# Static definitions surfaced in the dashboard glossary panel.
GLOSSARY = sorted([
    {"id": "evaluate-extracted-string", "term": "evaluate_extracted_string", "def": "Checks one short answer, such as a date or newspaper name. It gives some credit when the wording is close and correctly handles answers that are missing."},
    {"id": "evaluate-list", "term": "evaluate_list", "def": "Checks an answer that contains several items. It looks at which items are present and whether they are in the expected order."},
    {"id": "evaluate-raw-string", "term": "evaluate_raw_string", "def": "Checks a longer written answer by comparing its words with the words in the expected answer."},
    {"id": "framework", "term": "framework", "def": "The software or online service that runs the model doing the review."},
    {"id": "gpu-type", "term": "gpu_type", "def": "The kind of graphics processor used to run a model on our own computers. Online model services do not share this information."},
    {"id": "llm-time-total", "term": "llm_time_total", "def": "The total time spent waiting for the reviewing model to answer. For an online service, this also includes network delays and time spent waiting in the service's queue."},
    {"id": "metric-identified", "term": "Metric identified — selection_accuracy", "def": "Whether the agent said it chose the right way to check each answer."},
    {"id": "misrouted", "term": "Misrouted (paths table)", "def": "Runs where the agent chose the wrong way to check an answer. These are confirmed mistakes, not missing results."},
    {"id": "optimal-route", "term": "Optimal route — routing_path_correct", "def": "Whether the agent followed the entire correct process: load the answer once, check every part in the right way, and save the result once."},
    {"id": "overhead-time", "term": "overhead_time", "def": "Time spent running the agent and its checking tools, apart from time spent waiting for model replies."},
    {"id": "prompt", "term": "prompt", "def": "The instructions given to the agent doing the review. They are different from the instructions originally given to the model whose answer is being checked."},
    {"id": "save-success", "term": "save_success", "def": "The percentage of runs in which the agent successfully recorded its final result."},
    {"id": "score-consistent", "term": "score_consistent", "def": "Whether the final score the agent recorded agrees with the detailed scores produced during the same run."},
    {"id": "steps", "term": "steps", "def": "The number of times the agent asked the model what to do next before the task ended."},
    {"id": "stopped-reason", "term": "stopped_reason", "def": "Why the run ended: the agent finished, reached the allowed number of steps, or encountered a problem."},
    {"id": "total-dollar-cost", "term": "total_dollar_cost", "def": "The estimated cost in US dollars recorded when this run happened. Models run on our own computers cost $0 here; runs without a known price are shown as unpriced."},
    {"id": "total-tokens", "term": "total_tokens", "def": "The total amount of text the reviewing model read and wrote across all steps. A token is a small piece of a word or sentence."},
    {"id": "tokens-per-sec", "term": "tokens_per_sec", "def": "How quickly the reviewing model wrote its answer, measured in small pieces of text per second. It does not count the text the model read."},
    {"id": "unscored", "term": "Unscored (paths table)", "def": "Runs for which no final result could be checked, usually because the agent did not save one or the saved answer could not be matched to the expected task."},
    {"id": "wall-time-total", "term": "wall_time_total", "def": "The total time from the beginning to the end of a run, including model response time and work done by the agent's tools."},
], key=lambda item: item["term"].casefold())


def _present_columns(con: Any) -> set[str]:
    """Column names actually available in the runs view (older Parquet may lack new ones)."""
    return {d[0] for d in con.execute("SELECT * FROM runs LIMIT 0").description}


def build_prompt_registry(prompt_names: set[str] | list[str]) -> dict[str, dict[str, str]]:
    """{prompt_name: {system, user}} for the prompt versions we can source.

    Resolved per name from the agent_eval/prompts registry, so EVERY variant still on
    disk renders its real text (comparing v1 vs v2 in the UI needs both). A name the
    registry doesn't know — a variant since renamed or deleted — is omitted, and the UI
    shows a "(not embedded)" placeholder; persisting prompt text per run at run time is
    the follow-up for full historical fidelity.
    """
    registry = {}
    for name in prompt_names:
        try:
            _, system, _, _ = resolve_prompt(name)
            user = eval_user_prompt("<task_id>", "<run_id>", prompt=name)
        except KeyError:
            continue
        registry[name] = {"system": system.strip(), "user": user}
    return registry


# ── tool-path LLM summaries ────────────────────────────────────────────

def _path_signature(tool_sequence_json: str | None) -> str:
    """Render a run's tool_sequence_json as a ' → '-joined path signature ('' if unparseable)."""
    try:
        seq = json.loads(tool_sequence_json or "[]")
    except (TypeError, json.JSONDecodeError):
        return ""
    return " → ".join(seq)


def _load_cache(path: str | Path) -> dict[str, str]:
    """Load the path-summary cache from `path` ({} if it is absent or corrupt)."""
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_cache(path: str | Path, cache: dict[str, str]) -> None:
    """Persist the path-summary cache to `path`, creating parent dirs as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache))


def _summary_messages(signature: str, reasonings: list[str]) -> list[dict[str, str]]:
    """Build the chat messages asking for a one-sentence 'why this path' explanation."""
    joined = "\n---\n".join(reasonings[:3]) if reasonings else "(no reasoning captured)"
    return [
        {"role": "system", "content":
            "You explain, in ONE concise sentence, why an evaluation agent took a "
            "given tool-call path. The agent classifies each field's data shape and "
            "routes it to one composite type-tool (evaluate_raw_string / "
            "evaluate_extracted_string / evaluate_list). If a path routes a field to "
            "the wrong type-tool, say so plainly."},
        {"role": "user", "content":
            f"Tool-call path: {signature}\n\nAgent reasoning samples:\n{joined}\n\n"
            "One sentence on why the agent took this path."},
    ]


def _summarize_one(
    client: Any, model: str, completion_kwargs: dict[str, Any],
    signature: str, reasonings: list[str],
) -> str | None:
    """One summary call; returns text or None on any failure (graceful skip)."""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=_summary_messages(signature, reasonings),
            **completion_kwargs,
        )
        return (resp.choices[0].message.content or "").strip() or None
    except Exception:  # noqa: BLE001 — a down/erroring summarizer must not fail the build
        return None


def summarize_paths(
    con: Any, present: set[str], summarizer_backend: str, top_n: int = 6,
    cache_path: str | Path = DEFAULT_CACHE, refresh: bool = False,
) -> dict[str, str]:
    """Return {path_signature: blurb} for the top-N tool paths.

    Summaries are cached by PATH SIGNATURE in a JSON file, so a path is summarized
    once and reused across builds — later builds make zero LLM calls unless a new
    path appears (or refresh=True forces regeneration). The blurb explains the
    path itself, so it stays valid as more runs of that path accumulate. Returns
    {} — never raises — if the needed columns are missing or the summarizer is
    unreachable.
    """
    if not {"tool_sequence_json", "reasoning_json", "prompt_name"} <= present:
        return {}

    df = con.execute(
        """SELECT tool_sequence_json, reasoning_json
           FROM runs
           WHERE starts_with(prompt_name, ?)""",
        [METRIC_EVAL_PROMPT_PREFIX],
    ).df()

    groups: dict = {}
    for _, r in df.iterrows():
        sig = _path_signature(r["tool_sequence_json"])
        if not sig:
            continue
        g = groups.setdefault(sig, {"n": 0, "reasonings": []})
        g["n"] += 1
        rj = r["reasoning_json"]
        if rj and len(g["reasonings"]) < 3:
            g["reasonings"].append(str(rj)[:1500])
    if not groups:
        return {}

    top = sorted(groups.items(), key=lambda kv: kv[1]["n"], reverse=True)[:top_n]
    cache = {} if refresh else _load_cache(cache_path)

    out, misses = {}, []
    for sig, g in top:
        if sig in cache:               # summary already exists -> reuse, no LLM call
            out[sig] = cache[sig]
        else:
            misses.append((sig, g))

    if misses:
        try:
            client, model, ckwargs = config.sync_openai_client(summarizer_backend, timeout=15)
        except Exception:  # noqa: BLE001 — unknown/unbuildable summarizer -> cache-only
            return out
        for sig, g in misses:
            blurb = _summarize_one(client, model, ckwargs, sig, g["reasonings"])
            if blurb:
                out[sig] = blurb
                cache[sig] = blurb
        _save_cache(cache_path, cache)

    return out


# ── snapshot + render ──────────────────────────────────────────────────

def _version_dates(commits: set[str | None]) -> dict[str, str]:
    """Map each git_commit to its commit date (YYYY-MM-DD) via `git show`.

    Best-effort: a `-dirty` suffix is stripped to the hash; unknown or
    uncomputable commits are simply omitted (the UI falls back to the bare hash).
    """
    out: dict[str, str] = {}
    for c in commits:
        if not c:
            continue
        h = c.split("-", 1)[0]  # strip a "-dirty" suffix
        try:
            r = subprocess.run(
                ["git", "show", "-s", "--format=%cs", h],
                cwd=REPO_ROOT, capture_output=True, text=True, timeout=5,
            )
        except Exception:
            continue
        if r.returncode == 0 and r.stdout.strip():
            out[c] = r.stdout.strip()
    return out


def _version_metadata(commits: set[str | None]) -> dict[str, dict[str, Any]]:
    """Reader-facing labels for stored code versions, without losing the raw key."""
    dates = _version_dates(commits)
    out: dict[str, dict[str, Any]] = {}
    for commit in commits:
        if not commit:
            continue
        dirty = commit.endswith("-dirty")
        clean = commit[:-6] if dirty else commit
        short = clean[:7]
        date = dates.get(commit)
        if date:
            pretty = datetime.strptime(date, "%Y-%m-%d").strftime("%b %d, %Y").replace(" 0", " ")
            label = f"{pretty} · code {short}"
        else:
            label = f"Code {short}"
        if dirty:
            label += " · modified"
        out[commit] = {
            "short_sha": short,
            "commit_date": date,
            "dirty": dirty,
            "label": label,
        }
    return out


def build_snapshot(
    base_dir: str | Path | None = None, summarizer_backend: str = "summarizer",
    summarize: bool = True, top_n: int = 6, cache_path: str | Path = DEFAULT_CACHE,
    refresh_summaries: bool = False,
) -> dict[str, Any]:
    """Query DuckDB and return a JSON-serializable snapshot dict."""
    con = connect(base_dir)
    present = _present_columns(con)
    cols = [c for c in RUN_COLUMNS if c in present]
    if "prompt_name" in present:
        df = con.execute(
            f"SELECT {', '.join(cols)} FROM runs WHERE starts_with(prompt_name, ?)",
            [METRIC_EVAL_PROMPT_PREFIX],
        ).df()
    else:
        # Runs predating prompt identity cannot be assigned to a task safely.
        df = con.execute(f"SELECT {', '.join(cols)} FROM runs WHERE FALSE").df()
    # to_json -> loads gives native JSON types with NaN -> null.
    runs = json.loads(df.to_json(orient="records"))
    # Stable shape: any RUN_COLUMNS missing from older Parquet come back as null.
    for r in runs:
        for c in RUN_COLUMNS:
            r.setdefault(c, None)

    prompt_names = {r.get("prompt_name") for r in runs if r.get("prompt_name")}

    path_summaries = {}
    if summarize and summarizer_backend:
        path_summaries = summarize_paths(
            con, present, summarizer_backend, top_n, cache_path, refresh=refresh_summaries
        )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_runs": len(runs),
        "runs": runs,
        "prompts": build_prompt_registry(prompt_names),
        "glossary": GLOSSARY,
        "path_summaries": path_summaries,
        "type_tools": TYPE_TOOLS,
        "version_dates": _version_dates({r.get("git_commit") for r in runs}),
        "version_metadata": _version_metadata({r.get("git_commit") for r in runs}),
        "pricing": config.BACKENDS.get("playground", {}).get("pricing", {}),
    }


def render_html(snapshot: dict[str, Any], template_path: str | Path = TEMPLATE) -> str:
    """Inject the snapshot JSON into the template, escaping `<` so an embedded
    `</script>` can't break out of the data block."""
    template = Path(template_path).read_text()
    data_json = json.dumps(snapshot).replace("<", "\\u003c")
    if PLACEHOLDER not in template:
        raise ValueError(f"template missing placeholder {PLACEHOLDER}")
    return template.replace(PLACEHOLDER, data_json)


def main(argv: list[str] | None = None) -> None:
    """CLI entry: build the snapshot, render the template, and write the HTML file."""
    p = argparse.ArgumentParser(prog="analysis.build_dashboard")
    p.add_argument("--base-dir", default=None, help="override outputs/agent_runs")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--open", action="store_true", help="open the file in a browser")
    p.add_argument("--summarizer-backend", default="summarizer",
                   help="backend used to write tool-path summaries (must be reachable at build time)")
    p.add_argument("--summary-top-n", type=int, default=6, help="how many top paths to summarize")
    p.add_argument("--no-summaries", action="store_true", help="skip LLM path summaries")
    p.add_argument("--refresh-summaries", action="store_true",
                   help="regenerate cached path summaries instead of reusing them")
    args = p.parse_args(argv)

    snapshot = build_snapshot(
        args.base_dir,
        summarizer_backend=args.summarizer_backend,
        summarize=not args.no_summaries,
        top_n=args.summary_top_n,
        refresh_summaries=args.refresh_summaries,
    )
    html = render_html(snapshot)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    n_summaries = len(snapshot.get("path_summaries", {}))
    print(f"Wrote {out}  ({snapshot['n_runs']} runs, {n_summaries} path summaries)")
    if args.open:
        webbrowser.open(out.resolve().as_uri())


if __name__ == "__main__":
    main()
