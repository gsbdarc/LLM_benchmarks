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

try:
    import weave as _weave
except Exception:  # weave optional (e.g. in unit tests)
    _weave = None


def op(fn=None):
    """Decorator: become `weave.op()` unless weave is unavailable or disabled.

    Set EVAL_DISABLE_WEAVE=1 to make every decorated function a plain function —
    used by the test suite so unit tests never touch the network or Weave.
    Usage: `@op` (no parentheses).
    """
    def wrap(f):
        if _weave is not None and os.getenv("EVAL_DISABLE_WEAVE") != "1":
            return _weave.op()(f)
        return f

    return wrap(fn) if fn is not None else wrap

# ---------------------------------------------------------------------------
#  Printing / formatting (from notebook)
# ---------------------------------------------------------------------------

def log(section, message=""):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"\n[{ts}] {section}")
    if message:
        print(message)


def pretty(obj):
    return json.dumps(obj, indent=2, default=str)


def preview_text(text, max_chars=1200):
    if text is None:
        return ""
    text = str(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n... [truncated {len(text) - max_chars} chars]"


# ---------------------------------------------------------------------------
#  Usage / reasoning extraction (from notebook)
# ---------------------------------------------------------------------------

def usage_to_dict(response):
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


def mcp_result_to_text(result):
    return "\n".join(
        item.text for item in result.content if getattr(item, "type", None) == "text"
    )


THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def split_thinking(content):
    """Extract <think>...</think> blocks (NIM/Gemma). Returns (reasoning, visible_content)."""
    if not content:
        return "", content or ""
    reasoning = "\n".join(m.strip() for m in THINK_RE.findall(content))
    visible = THINK_RE.sub("", content).strip()
    return reasoning, visible


def extract_reasoning(msg, response):
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

def compute_prompt_hash(text):
    """First 8 hex chars of sha256 of the system prompt — stable ID for prompt variants."""
    return hashlib.sha256(text.encode()).hexdigest()[:8]


def compute_tools_hash(tool_schemas):
    """sha256 of the JSON-serialized MCP tool schema list — detects server drift."""
    return hashlib.sha256(json.dumps(tool_schemas, sort_keys=True).encode()).hexdigest()[:8]


def get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
#  New: per-run performance derivations (plan §2)
# ---------------------------------------------------------------------------

def tokens_per_second(completion_tokens, seconds):
    """completion tokens / generation time. None if time is missing/zero."""
    if not seconds or seconds <= 0:
        return None
    if completion_tokens is None:
        return None
    return completion_tokens / seconds


def tool_sequence(steps_detail):
    """Ordered list of tool names as the agent called them, flattened across steps.

    Drives the tool-call path visualization (which path did a run take, and how
    many runs took it). Within a step, calls keep the order the model emitted.
    """
    seq = []
    for s in steps_detail or []:
        for name in (s.get("tool_calls") or []):
            seq.append(name)
    return seq


def peak_context(steps_detail):
    """Largest prompt_tokens seen across steps — the peak context the model held."""
    vals = []
    for s in steps_detail or []:
        u = s.get("usage") or {}
        v = u.get("prompt_tokens")
        if isinstance(v, (int, float)):
            vals.append(v)
    return max(vals) if vals else None


def reasoning_blob(steps_detail, max_chars=4000):
    """Compact, bounded per-run reasoning for the dashboard path-summaries.

    Returns a JSON string of [{"step", "reasoning", "tool_calls"}] for steps that
    produced any reasoning (NIM <think> or API reasoning), truncated to a total of
    ~max_chars so one wordy run can't bloat the Parquet row. None if no reasoning
    was captured. This is the persisted input Part 4's LLM path-summaries read.
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


def client_ram_mb():
    """RSS of THIS (client) process in MB. NOT GPU RAM — that comes from nvidia-smi."""
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        return None


# ---------------------------------------------------------------------------
#  New: vLLM / NIM Prometheus /v1/metrics scrape (plan §2)
# ---------------------------------------------------------------------------

def parse_prometheus(text):
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


def select_vllm_metrics(parsed):
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


def scrape_vllm_metrics(metrics_url, timeout=5.0):
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
