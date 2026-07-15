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
import webbrowser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from analysis.queries import connect
from agent_eval import config
from agent_eval.prompts import METRIC_EVAL_SYSTEM, PROMPT_NAME, eval_user_prompt

REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = Path(__file__).resolve().parent / "dashboard_template.html"
DEFAULT_OUT = REPO_ROOT / "images" / "agent_dashboard.html"
PLACEHOLDER = "__AGENTVIZ_DATA__"
DEFAULT_CACHE = REPO_ROOT / "outputs" / ".path_summary_cache.json"

# The three composite type-tools (for the routing / glossary views).
TYPE_TOOLS = ["evaluate_raw_string", "evaluate_extracted_string", "evaluate_list"]

# Columns embedded per run. Kept explicit so the payload is stable and small.
# NOTE: reasoning_json is intentionally NOT embedded per-run (it can be large);
# it is read separately at build time to generate the path summaries.
RUN_COLUMNS = [
    "eval_id", "task_id", "run_id", "benchmark_id", "model_id",
    "backend", "framework", "model", "agent_model_key",
    "temperature", "gpu_type", "reasoning_level", "prompt_name", "prompt_key", "prompt_hash",
    "concurrency", "steps", "stopped_reason",
    "n_tool_calls", "n_metric_calls", "n_tool_errors", "tool_sequence_json",
    "prompt_tokens", "completion_tokens", "total_tokens",
    "tokens_per_sec", "peak_context", "wall_time_total", "llm_time_total", "overhead_time",
    "save_success", "save_count", "save_failed", "score_consistent", "selection_accuracy",
    "gpu_cache_usage_end", "requests_running_end", "weave_trace_url",
]

# Static definitions surfaced in the dashboard glossary panel.
GLOSSARY = [
    {"term": "save_success", "def": "Fraction of runs where the agent successfully wrote its evaluation via save_evaluation."},
    {"term": "score_consistent", "def": "The composite score the agent SAVED matches the score recomputed from the tool's own sub-scores (integrity check)."},
    {"term": "selection_accuracy / routing acc", "def": "Did the agent route the field to the CORRECT composite type-tool, graded against the gold field_type."},
    {"term": "evaluate_raw_string", "def": "Composite tool for free-form prose fields. Scores with word-IoU."},
    {"term": "evaluate_extracted_string", "def": "Composite tool for a single extracted value that may be absent. null_accuracy × max(levenshtein, char_f1)."},
    {"term": "evaluate_list", "def": "Composite tool for set/sequence fields. max(set_f1, sequence_lcs, set_inclusion)."},
    {"term": "llm_time_total", "def": "Sum of the agent's LLM request round-trips (model/service latency). For remote endpoints (playground) it includes network + shared-API queueing, so read it as SERVICE latency, not pure inference; only local models approximate inference time."},
    {"term": "overhead_time", "def": "wall_time_total − llm_time_total ≈ tool-execution + agent-loop time. The MCP tool server is local for all backends, so this is backend-independent."},
    {"term": "wall_time_total", "def": "End-to-end agent time = llm_time_total + overhead_time (excludes the GPU-metrics scrapes, which are timed outside it)."},
    {"term": "total_tokens", "def": "prompt + completion tokens summed across ALL agent steps. Each step re-sends the growing conversation as its prompt, so this is ~95% prompt tokens and far larger than tokens_per_sec × time (which reconstructs only generated tokens)."},
    {"term": "tokens_per_sec", "def": "completion (generated) tokens per second of llm_time — generation throughput. Excludes prompt tokens, so tokens_per_sec × llm_time ≈ completion tokens, NOT total_tokens."},
    {"term": "steps", "def": "Number of agent turns (tool-call rounds) taken to finish a task."},
    {"term": "stopped_reason", "def": "Why the agent loop ended: answered, max_steps, or error."},
    {"term": "prompt", "def": "The AGENT's system+user prompt (versioned by prompt_name) — NOT the original benchmark prompt given to the model under test."},
    {"term": "framework", "def": "Serving engine behind the model endpoint (vllm, nim, ollama, openai). Determines whether concurrent requests batch."},
    {"term": "gpu_type", "def": "GPU model of the server hosting the model under test (from nvidia-smi on the server host)."},
]


def _present_columns(con: Any) -> set[str]:
    """Column names actually available in the runs view (older Parquet may lack new ones)."""
    return {d[0] for d in con.execute("SELECT * FROM runs LIMIT 0").description}


def build_prompt_registry(prompt_names: set[str] | list[str]) -> dict[str, dict[str, str]]:
    """{prompt_name: {system, user}} for the prompt versions we can source.

    Seeded from the agent_eval/prompts registry, so the CURRENT prompt renders its real text.
    Historical prompt versions whose text we don't have are omitted (the UI shows
    a "(not embedded)" placeholder); persisting prompt text per version over time
    is the follow-up for full historical fidelity.
    """
    registry = {}
    for name in prompt_names:
        if name == PROMPT_NAME:
            registry[name] = {
                "system": METRIC_EVAL_SYSTEM.strip(),
                "user": eval_user_prompt("<task_id>", "<run_id>"),
            }
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
    if not {"tool_sequence_json", "reasoning_json"} <= present:
        return {}

    df = con.execute("SELECT tool_sequence_json, reasoning_json FROM runs").df()

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

def build_snapshot(
    base_dir: str | Path | None = None, summarizer_backend: str = "summarizer",
    summarize: bool = True, top_n: int = 6, cache_path: str | Path = DEFAULT_CACHE,
    refresh_summaries: bool = False,
) -> dict[str, Any]:
    """Query DuckDB and return a JSON-serializable snapshot dict."""
    con = connect(base_dir)
    present = _present_columns(con)
    cols = [c for c in RUN_COLUMNS if c in present]
    df = con.execute(f"SELECT {', '.join(cols)} FROM runs").df()
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
