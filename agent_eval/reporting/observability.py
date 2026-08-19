"""
observability.py — instrumentation helpers.

Two groups:
  1. Lifted verbatim from the notebook helpers cell (be3af12f): printing,
     usage/reasoning extraction, and the prompt/tools/git hashes.
  2. New for this work (plan §2): per-run derivations (tokens/sec, peak context)
     and a scrape of the vLLM/NIM Prometheus `/v1/metrics` endpoint for GPU
     cache + request-queue signals. GPU SM% and RAM come from the nvidia-smi
     sidecar in the SLURM job, joined later by timestamp — they cannot be read
     from the client because the model runs on a different node.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from datetime import datetime
from typing import Any, Callable, Optional

try:
    import weave as _weave
except Exception:  # weave optional (e.g. in unit tests)
    _weave = None


def op(fn: Optional[Callable[..., Any]] = None) -> Callable[..., Any]:
    """Decorator: become `weave.op()` unless weave is unavailable or disabled.

    Set EVAL_DISABLE_WEAVE=1 to make every decorated function a plain function —
    used by the test suite so unit tests never touch the network or Weave.
    Usage: `@op` (no parentheses).
    """
    def wrap(f: Callable[..., Any]) -> Callable[..., Any]:
        if _weave is not None and os.getenv("EVAL_DISABLE_WEAVE") != "1":
            return _weave.op()(f)
        return f

    return wrap(fn) if fn is not None else wrap


def current_trace_url() -> Optional[str]:
    """Weave UI URL of the currently-executing op call, else None.

    Call it from inside an @op (e.g. run_agent) so it resolves that run's own
    trace. Returns None when Weave is unavailable/disabled or no call is active —
    so the dashboard's per-run link is populated on real runs and null in tests.
    """
    if _weave is None or os.getenv("EVAL_DISABLE_WEAVE") == "1":
        return None
    try:
        call = _weave.get_current_call()
        return getattr(call, "ui_url", None)
    except Exception:  # noqa: BLE001 — never fail a run over a missing trace URL
        return None

# ---------------------------------------------------------------------------
#  Printing / formatting (from notebook)
# ---------------------------------------------------------------------------

def log(section: str, message: str = "") -> None:
    """Print a timestamped section header, plus an optional message body."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"\n[{ts}] {section}")
    if message:
        print(message)


def pretty(obj: Any) -> str:
    """Pretty-print any object as indented JSON (falling back to str)."""
    return json.dumps(obj, indent=2, default=str)


def preview_text(text: Any, max_chars: int = 1200) -> str:
    """Return text truncated to max_chars with a note of how much was dropped."""
    if text is None:
        return ""
    text = str(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n... [truncated {len(text) - max_chars} chars]"


# ---------------------------------------------------------------------------
#  Usage / reasoning extraction (from notebook)
# ---------------------------------------------------------------------------

def usage_to_dict(response: Any) -> Optional[dict[str, Any]]:
    """Extract token usage from a response as a dict, or None if absent."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def mcp_result_to_text(result: Any) -> str:
    """Join the text parts of an MCP tool result into a single string."""
    return "\n".join(
        item.text for item in result.content if getattr(item, "type", None) == "text"
    )


THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def split_thinking(content: Optional[str]) -> tuple[str, str]:
    """Extract <think>...</think> blocks (NIM/Gemma). Returns (reasoning, visible_content)."""
    if not content:
        return "", content or ""
    reasoning = "\n".join(m.strip() for m in THINK_RE.findall(content))
    visible = THINK_RE.sub("", content).strip()
    return reasoning, visible


def extract_reasoning(msg: Any, response: Any) -> Any:
    """Pull reasoning from message.reasoning_content / message.reasoning / response.reasoning."""
    r = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None)
    if r is None:
        r = getattr(response, "reasoning", None)
    if r and hasattr(r, "model_dump"):
        r = r.model_dump()
    return r


# ---------------------------------------------------------------------------
#  Stable IDs (from notebook)
# ---------------------------------------------------------------------------

def compute_prompt_hash(text: str) -> str:
    """First 8 hex chars of sha256 of the system prompt — stable ID for prompt variants."""
    return hashlib.sha256(text.encode()).hexdigest()[:8]


def compute_tools_hash(tool_schemas: Any) -> str:
    """sha256 of the JSON-serialized MCP tool schema list — detects server drift."""
    return hashlib.sha256(json.dumps(tool_schemas, sort_keys=True).encode()).hexdigest()[:8]


def get_git_commit() -> str:
    """Short git commit hash of HEAD, suffixed '-dirty' when the working tree differs
    from HEAD, or "unknown" if git is unavailable.

    Runs are versioned by this value (code_version): the Mongo stores key on it so a
    re-run under a new commit coexists with old results instead of overwriting them.
    The '-dirty' flag matters because a dirty tree means the recorded version does not
    fully pin the code — commit before a versioned run for a clean code_version.
    """
    try:
        h = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"
    try:
        dirty = subprocess.call(
            ["git", "diff", "--quiet", "HEAD"],
            stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
        ) != 0
    except Exception:
        dirty = False
    return h + ("-dirty" if dirty else "")


# ---------------------------------------------------------------------------
#  New: per-run performance derivations (plan §2)
# ---------------------------------------------------------------------------

def tokens_per_second(
    completion_tokens: Optional[float], seconds: Optional[float]
) -> Optional[float]:
    """completion tokens / generation time. None if time is missing/zero."""
    if not seconds or seconds <= 0:
        return None
    if completion_tokens is None:
        return None
    return completion_tokens / seconds


def tool_sequence(steps_detail: Optional[list[dict[str, Any]]]) -> list[str]:
    """Ordered list of tool names as the agent called them, flattened across steps.

    Drives the tool-call path visualization (which path did a run take, and how
    many runs took it). Within a step, calls keep the order the model emitted.
    """
    seq = []
    for s in steps_detail or []:
        for name in (s.get("tool_calls") or []):
            seq.append(name)
    return seq


def peak_context(steps_detail: Optional[list[dict[str, Any]]]) -> Optional[float]:
    """Largest prompt_tokens seen across steps — the peak context the model held."""
    vals = []
    for s in steps_detail or []:
        u = s.get("usage") or {}
        v = u.get("prompt_tokens")
        if isinstance(v, (int, float)):
            vals.append(v)
    return max(vals) if vals else None


def reasoning_tokens(steps_detail: Optional[list[dict[str, Any]]]) -> Optional[int]:
    """Total reasoning tokens across steps, when the provider reports them.

    `usage_to_dict` keeps the full usage payload, so this reads
    completion_tokens_details.reasoning_tokens. The gpt-5/o1 family bills reasoning
    without ever returning the text, so this is the only signal of how much thinking
    happened there. None when no step reported it.
    """
    total = None
    for s in steps_detail or []:
        details = (s.get("usage") or {}).get("completion_tokens_details") or {}
        v = details.get("reasoning_tokens")
        if isinstance(v, (int, float)):
            total = (total or 0) + int(v)
    return total


def steps_trace(
    steps_detail: Optional[list[dict[str, Any]]],
    max_steps: int = 24,
    max_chars: int = 2000,
) -> Optional[str]:
    """Per-step trace of what the agent did, as a JSON string. None if no steps.

    This is the record error analysis reads: for each step, the prose the model
    emitted, the tool calls WITH their raw argument strings, and that step's latency
    and tokens. Arguments are kept unparsed so malformed tool JSON — a real failure
    mode here — stays visible instead of being normalised away.

    Bounded (default ~24 steps, ~2000 prose chars each ≈ 12 KB/run) so one runaway
    run cannot bloat the row; `truncated` marks where that happened.
    """
    if not steps_detail:
        return None
    out = []
    for s in steps_detail[:max_steps]:
        visible = s.get("visible")
        if isinstance(visible, str) and len(visible) > max_chars:
            visible, clipped = visible[:max_chars] + "…", True
        else:
            clipped = False
        usage = s.get("usage") or {}
        entry = {
            "step": s.get("step"),
            "finish_reason": s.get("finish_reason"),
            "visible": visible,
            "tool_calls": s.get("tool_call_args") or [],
            "llm_time": s.get("llm_time"),
            "llm_time_productive": s.get("llm_time_productive"),
            "attempts": s.get("attempts"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
        }
        for key in ("thinking_api", "thinking_nim", "retry_errors"):
            if s.get(key):
                entry[key] = s[key]
        if clipped:
            entry["truncated"] = True
        out.append(entry)
    if len(steps_detail) > max_steps:
        out.append({"note": f"{len(steps_detail) - max_steps} further steps omitted"})
    return json.dumps(out)


def reasoning_blob(
    steps_detail: Optional[list[dict[str, Any]]], max_chars: int = 4000
) -> Optional[str]:
    """Compact, bounded per-run reasoning for the dashboard path-summaries.

    Captures PROVIDER reasoning only (NIM <think> blocks or an API reasoning field),
    so it is None for models that don't expose one — which is every model configured
    before gemini-2.5-pro, hence the null `reasoning_json` on older rows. It is not
    broken; there was simply nothing to record. `steps_trace` above is the record that
    always exists.
    """
    out = []
    budget = max_chars
    for s in steps_detail or []:
        text = s.get("thinking_nim") or s.get("thinking_api")
        if not text:
            continue
        text = str(text)
        if len(text) > budget:
            text = text[:budget] + "…"
        out.append({
            "step": s.get("step"),
            "reasoning": text,
            "tool_calls": s.get("tool_calls") or [],
        })
        budget -= len(text)
        if budget <= 0:
            break
    return json.dumps(out) if out else None


def client_ram_mb() -> Optional[float]:
    """RSS of THIS (client) process in MB. NOT GPU RAM — that comes from nvidia-smi."""
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        return None


# ---------------------------------------------------------------------------
#  New: vLLM / NIM Prometheus /v1/metrics scrape (plan §2)
# ---------------------------------------------------------------------------

def parse_prometheus(text: str) -> dict[str, float]:
    """Parse Prometheus text exposition into {metric_family_name: summed_value}.

    Labels are stripped and values summed across label sets. For our single-model
    server each family has one series, so the sum is just that value. Counters
    (…_total) and gauges (cache %, queue depth) are all returned as floats.
    """
    out: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            name_part, value = line.rsplit(" ", 1)
        except ValueError:
            continue
        try:
            val = float(value)
        except ValueError:
            continue
        name = name_part.split("{", 1)[0].strip()
        out[name] = out.get(name, 0.0) + val
    return out


# vLLM prefixes families with "vllm:"; some builds drop it. Try both.
_VLLM_KEYS = {
    "gpu_cache_usage_perc": ("vllm:gpu_cache_usage_perc", "gpu_cache_usage_perc"),
    "num_requests_running": ("vllm:num_requests_running", "num_requests_running"),
    "num_requests_waiting": ("vllm:num_requests_waiting", "num_requests_waiting"),
    "prompt_tokens_total": ("vllm:prompt_tokens_total", "prompt_tokens_total"),
    "generation_tokens_total": ("vllm:generation_tokens_total", "generation_tokens_total"),
}


def select_vllm_metrics(parsed: dict[str, float]) -> dict[str, Optional[float]]:
    """Pick the GPU/queue/throughput signals we care about from a parsed dump."""
    result = {}
    for out_key, candidates in _VLLM_KEYS.items():
        value = None
        for c in candidates:
            if c in parsed:
                value = parsed[c]
                break
        result[out_key] = value
    return result


def scrape_vllm_metrics(metrics_url: str, timeout: float = 5.0) -> dict[str, Any]:
    """GET the Prometheus endpoint and return the selected metrics.

    Returns the selected-metrics dict on success, or {"error": ...} on any
    failure (server down, wrong URL, timeout) — never raises, so a missing
    metrics endpoint can't break an evaluation run.
    """
    try:
        import requests

        resp = requests.get(metrics_url, timeout=timeout)
        resp.raise_for_status()
        return select_vllm_metrics(parse_prometheus(resp.text))
    except Exception as e:  # noqa: BLE001 — surface as data, never crash a run
        return {"error": str(e)}
