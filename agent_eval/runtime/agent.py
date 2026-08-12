"""
agent.py — MCP tool discovery and the agent loop.

Two changes from the notebook version (plan §4):

  1. Persistent MCP session. The notebook opened a fresh streamable-HTTP
     connection on EVERY tool call (call_mcp_tool). Under parallel load that
     connection churn is the prime suspect for the dropped connections that
     forced Semaphore(2). Here, run_agent opens ONE session for its whole loop
     and stashes it in a contextvar, so N concurrent agents hold N sessions, not
     N×(tools-per-run) short-lived ones. The contextvar (rather than a function
     arg) keeps the session out of Weave's traced inputs.

  2. Retry/backoff on connection-reset style errors for both the MCP tool call
     and the LLM completion, so a transient drop self-heals instead of failing
     the run.

Everything else (reasoning extraction, per-step detail, usage tallies) is lifted
from run_agent in the notebook (cell 921bc37f), plus the new per-run derivations
(tokens/sec, peak context, optional GPU scrape) from plan §2.
"""

from __future__ import annotations

import contextvars
import json
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Callable, Optional

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from ..reporting import observability as obs
from ..reporting.integrity import SAVE_TOOLS
from ..reporting.observability import op

# The active MCP session for the current asyncio task. asyncio.gather wraps each
# coroutine in its own Task with a copied context, so concurrent run_agent calls
# never see each other's session.
_session_var: contextvars.ContextVar = contextvars.ContextVar("mcp_session", default=None)


# ---------------------------------------------------------------------------
#  Retry policy
# ---------------------------------------------------------------------------

_RETRYABLE_NAMES = {
    # httpx / anyio connection + timeout errors
    "ConnectError", "ConnectionError", "ConnectionResetError", "ConnectTimeout",
    "ReadError", "ReadTimeout", "WriteError", "RemoteProtocolError",
    "TimeoutError", "TimeoutException", "ClosedResourceError", "BrokenResourceError",
    "EndOfStream", "PoolTimeout",
    # OpenAI SDK TRANSIENT API errors: rate-limit (429), transient server (5xx), timeouts.
    # Deliberately NOT BadRequestError/AuthenticationError/NotFoundError — those are permanent
    # (bad request / key / model), so they fail fast instead of wasting retries.
    "RateLimitError", "InternalServerError", "APITimeoutError", "APIConnectionError",
}


def _is_retryable(exc: BaseException) -> bool:
    """True for TRANSIENT errors worth retrying: connection/timeout AND
    rate-limit (429) / transient-server (5xx) API errors.

    Matches on the exception's class name (so we don't have to import httpx/anyio
    or the OpenAI SDK types) and on a transient substring in the message. Permanent
    errors (BadRequest/Authentication/NotFound) are absent from both → fail fast.
    """
    name = type(exc).__name__
    if name in _RETRYABLE_NAMES:
        return True
    msg = str(exc).lower()
    return any(s in msg for s in ("connection reset", "connection error", "timed out",
                                  "timeout", "broken pipe", "peer closed",
                                  "rate limit", "overloaded", "service unavailable",
                                  "temporarily unavailable"))


# 5 attempts / backoff to 30s: rate-limit (429) windows can need a longer wait than a
# connection blip. tenacity is the SINGLE retry layer (AsyncOpenAI is built with
# max_retries=0 in config.build_backend) so attempts don't multiply with the SDK's own.
_MAX_RETRY_AFTER = 120.0  # ignore an absurd header rather than stalling a batch


def _wait_honoring_retry_after(retry_state: Any) -> float:
    """Exponential backoff, but never shorter than a 429's `Retry-After`.

    Blind exponential backoff retries into a rate-limit window that the server has
    already told us the length of. When the header is present we wait at least that
    long (capped), otherwise this is exactly the previous behaviour.
    """
    base = wait_exponential(multiplier=0.5, min=0.5, max=30)(retry_state)
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    headers = getattr(getattr(exc, "response", None), "headers", None) or {}
    raw = headers.get("retry-after") or headers.get("Retry-After")
    try:
        after = float(raw)
    except (TypeError, ValueError):
        return base
    return max(base, min(after, _MAX_RETRY_AFTER))


_retry_policy = dict(
    retry=retry_if_exception(_is_retryable),
    stop=stop_after_attempt(5),
    wait=_wait_honoring_retry_after,
    reraise=True,
)


# ---------------------------------------------------------------------------
#  Session management
# ---------------------------------------------------------------------------

@asynccontextmanager
async def mcp_session(url: str) -> AsyncIterator[ClientSession]:
    """Open one initialized MCP ClientSession over streamable HTTP."""
    async with streamablehttp_client(url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


@asynccontextmanager
async def use_session(url: str) -> AsyncIterator[ClientSession]:
    """Open a session AND bind it to the contextvar for call_mcp_tool to find.

    Use this to make ad-hoc tool calls outside run_agent (e.g. the CLI calling
    list_outputs to build a dataset).
    """
    async with mcp_session(url) as session:
        token = _session_var.set(session)
        try:
            yield session
        finally:
            _session_var.reset(token)


# The server hosts tools for several tasks; a run should see only its own. Measured
# reason: `list_outputs` exists for runner.discover_rows to build the work list, but its
# description tells the model to "use this first" — of 360 stored runs, the 200 that
# called it saved less often (85% vs 100%) and cost ~75% more. Filtering is per-task
# rather than a blanket exclusion because the tool IS legitimate for discovery.
METRIC_EVAL_TOOLS = frozenset({
    "get_task_output", "save_evaluation",
    "evaluate_raw_string", "evaluate_extracted_string", "evaluate_list",
})
DATE_FIX_TOOLS = frozenset({"get_guide_date_case", "compute_guide_date", "save_correction"})

# prompt-name prefix -> the tools that task needs. A new task adds one entry.
_TOOLS_BY_TASK = (("date_fix", DATE_FIX_TOOLS),)


def tools_for_prompt(prompt_name: str | None) -> frozenset[str]:
    """The tool set a run should be shown, chosen by its prompt variant."""
    for prefix, allowed in _TOOLS_BY_TASK:
        if (prompt_name or "").startswith(prefix):
            return allowed
    return METRIC_EVAL_TOOLS


async def load_tools_from_mcp(url: str, allow: frozenset[str] | None = None) -> list[dict[str, Any]]:
    """Discover tools from the live server, as OpenAI Chat Completions schemas.

    `allow` restricts what the AGENT is shown; the server still serves everything and
    `call_mcp_tool` can still reach it, so discovery (runner.discover_rows) is unaffected.
    None means show every tool — the pre-existing behaviour.
    """
    async with mcp_session(url) as session:
        listed = await session.list_tools()
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description or "",
                "parameters": t.inputSchema,
            },
        }
        for t in listed.tools
        if allow is None or t.name in allow
    ]


# ---------------------------------------------------------------------------
#  Tool call + LLM step (both retried)
# ---------------------------------------------------------------------------

@retry(**_retry_policy)
async def _call_tool_retrying(session: ClientSession, tool_name: str, arguments: dict[str, Any]) -> Any:
    """Call one MCP tool with the retry policy applied to transient drops."""
    result = await session.call_tool(tool_name, arguments)
    return result


@op
async def call_mcp_tool(tool_name: str, arguments: dict[str, Any], verbose: bool = True) -> str:
    """Call one MCP tool over the session bound to the current context."""
    session = _session_var.get()
    if session is None:
        raise RuntimeError(
            "no MCP session in context — wrap calls in `async with use_session(url):` "
            "or run inside run_agent()"
        )
    if verbose:
        obs.log("MCP CALL", f"{tool_name}({obs.pretty(arguments)})")
    t0 = time.perf_counter()
    result = await _call_tool_retrying(session, tool_name, arguments)
    text = obs.mcp_result_to_text(result)
    dt = time.perf_counter() - t0
    if verbose:
        obs.log(
            "MCP RESULT",
            f"isError={getattr(result, 'isError', None)}  elapsed={dt:.2f}s  "
            f"chars={len(text)}\n{obs.preview_text(text)}",
        )
    return text


@op
def log_parse_failure(tool_name: str, raw_args: Optional[str], error_msg: str) -> dict[str, Any]:
    """Traced event for JSON argument-parse failures (behavior unchanged)."""
    return {"tool_name": tool_name, "raw_args": raw_args, "error": error_msg}


# Per-step attempt tally, written by llm_step and read by run_agent. A contextvar
# because make_llm_step's closure is shared across concurrent runs — a counter on the
# function (or tenacity's own retry.statistics) would mix runs together.
_attempts_var: contextvars.ContextVar = contextvars.ContextVar("llm_attempts", default=None)


def _record_attempt(seconds: float, ok: bool, error: Optional[str] = None) -> None:
    """Record one LLM attempt's duration and outcome, if a step is collecting."""
    tally = _attempts_var.get()
    if tally is not None:
        tally.append({"seconds": seconds, "ok": ok, **({"error": error} if error else {})})


def _summarize_attempts(tally: list[dict[str, Any]], step_total: float) -> dict[str, Any]:
    """Split a step's wall time into productive model latency vs retry overhead.

    Productive = the attempt that succeeded (the last one). Everything else — failed
    attempts and tenacity's backoff sleeps — is retry_wait. With no tally (a faked
    client in tests) productive falls back to the step total, so old numbers hold.
    """
    if not tally:
        return {"llm_time_productive": step_total, "llm_retry_wait": 0.0,
                "attempts": 1, "retry_errors": None}
    productive = next((a["seconds"] for a in reversed(tally) if a["ok"]), 0.0)
    errors = [a["error"] for a in tally if not a["ok"] and a.get("error")]
    return {
        "llm_time_productive": productive,
        "llm_retry_wait": max(0.0, step_total - productive),
        "attempts": len(tally),
        "retry_errors": ",".join(errors) if errors else None,
    }


def make_llm_step(
    client: Any,
    model: str,
    tools: list[dict[str, Any]],
    completion_kwargs: dict[str, Any],
) -> Callable[[list[Any]], Any]:
    """Build a retried, traced async llm_step bound to a client/model/tools/kwargs.

    Returned as a closure so run_agent stays backend-agnostic and tests can pass
    a fake client. `llm_step` is a COROUTINE: run_agent awaits it, so concurrent
    runs (asyncio.gather in runner.run_batch) truly overlap at the LLM layer
    instead of serializing on a blocking call. Retry covers transient connection
    resets (tenacity supports coroutine functions); @op traces async calls too.
    """

    @op
    @retry(**_retry_policy)
    async def llm_step(messages: list[Any]) -> Any:
        # Time each ATTEMPT individually. The caller's timer spans the whole retry
        # sequence, so without this the backoff sleep (up to 30 s x 5) is booked as
        # model latency. One closure is shared by every concurrent run, so the tally
        # lives in a contextvar rather than on the function object.
        t0 = time.perf_counter()
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                **completion_kwargs,
            )
        except BaseException as e:
            _record_attempt(time.perf_counter() - t0, ok=False, error=type(e).__name__)
            raise
        _record_attempt(time.perf_counter() - t0, ok=True)
        return response

    return llm_step


# ---------------------------------------------------------------------------
#  Agent loop
# ---------------------------------------------------------------------------

@op
async def run_agent(
    user_prompt: str,
    system_prompt: str,
    llm_step: Callable[[list[Any]], Any],
    mcp_url: str,
    max_steps: int = 12,
    verbose: bool = True,
    backend: Optional[str] = None,
    model: Optional[str] = None,
    task_id: Optional[str] = None,
    run_id: Optional[int] = None,
    metrics_url: Optional[str] = None,
    eval_id: Optional[int] = None,
    git_commit: Optional[str] = None,
) -> dict[str, Any]:
    """Run the agent on one output. Opens one MCP session for the whole loop.

    `llm_step` is the closure from make_llm_step. `metrics_url`, when given, is
    scraped once before and once after the run for GPU/queue context. `eval_id`, when
    given, is stamped onto the agent's save_evaluation call so the verdict is keyed
    per (output × judge) and joins the run-metrics row — set here client-side rather
    than trusting the LLM to relay it faithfully.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    tool_calls_by_name: dict = {}
    tool_time_by_name: dict = {}
    tool_errors_by_name: dict = {}
    steps_detail: list = []
    llm_time_total = 0.0
    llm_time_productive_total = 0.0
    llm_retry_wait_total = 0.0
    attempts_total = 0
    steps_run = 0
    stopped_reason = "max_steps"
    answer = "Stopped after maximum tool-calling steps."

    # Scrape GPU metrics BEFORE starting the wall clock (and again after stopping it,
    # below) so the scrape's own HTTP latency never lands inside wall_time_total. This
    # keeps wall_time backend-consistent: playground has no scrape, and NIM shouldn't
    # be penalised for one.
    gpu_start = obs.scrape_vllm_metrics(metrics_url) if metrics_url else None

    if verbose:
        obs.log("AGENT START", f"Model: {model}  Backend: {backend}")

    wall_t0 = time.perf_counter()
    async with use_session(mcp_url):
        for step in range(1, max_steps + 1):
            steps_run = step
            if verbose:
                obs.log(f"LLM CALL {step}", f"{len(messages)} messages")

            t0 = time.perf_counter()
            attempt_tally: list[dict[str, Any]] = []
            tally_token = _attempts_var.set(attempt_tally)
            try:
                response = await llm_step(messages)
            except Exception as e:  # noqa: BLE001
                _attempts_var.reset(tally_token)
                timing = _summarize_attempts(attempt_tally, time.perf_counter() - t0)
                llm_retry_wait_total += timing["llm_retry_wait"]
                attempts_total += timing["attempts"]
                stopped_reason = "error"
                answer = f"LLM error: {e}"
                if verbose:
                    obs.log("LLM ERROR", str(e))
                break
            _attempts_var.reset(tally_token)
            dt = time.perf_counter() - t0
            llm_time_total += dt
            timing = _summarize_attempts(attempt_tally, dt)
            llm_time_productive_total += timing["llm_time_productive"]
            llm_retry_wait_total += timing["llm_retry_wait"]
            attempts_total += timing["attempts"]

            usage = obs.usage_to_dict(response)
            if usage:
                for k in total_usage:
                    v = usage.get(k)
                    if isinstance(v, int):
                        total_usage[k] += v

            choice = response.choices[0]
            msg = choice.message
            messages.append(msg)

            raw_content = getattr(msg, "content", None)
            thinking_nim, visible = obs.split_thinking(raw_content)
            thinking_api = obs.extract_reasoning(msg, response)

            steps_detail.append({
                "step": step,
                "finish_reason": choice.finish_reason,
                "thinking_nim": thinking_nim or None,
                "thinking_api": str(thinking_api) if thinking_api else None,
                "tool_calls": [tc.function.name for tc in (msg.tool_calls or [])],
                # The model's own prose and the ARGUMENTS it chose — for these models
                # that is the whole reasoning record, and the arguments localise an
                # error faster than the prose (which often rationalises).
                # Raw argument strings, so malformed JSON stays visible.
                "visible": visible or None,
                "tool_call_args": [
                    {"name": tc.function.name, "arguments": tc.function.arguments}
                    for tc in (msg.tool_calls or [])
                ],
                "llm_time": dt,
                **timing,
                "usage": usage,
            })

            if verbose:
                if thinking_nim:
                    obs.log(f"REASONING (<think>) step {step}", obs.preview_text(thinking_nim, 2000))
                if thinking_api:
                    obs.log(f"REASONING (api) step {step}", obs.preview_text(str(thinking_api), 2000))
                obs.log(
                    f"LLM RESPONSE {step}",
                    f"finish={choice.finish_reason}  elapsed={dt:.2f}s\n{obs.preview_text(visible)}",
                )

            if not msg.tool_calls:
                stopped_reason = "answered"
                answer = msg.content
                if verbose:
                    obs.log("AGENT DONE", obs.pretty(total_usage))
                break

            for tc in msg.tool_calls:
                name = tc.function.name
                tool_calls_by_name[name] = tool_calls_by_name.get(name, 0) + 1
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError as e:
                    # Do NOT fall back to args={} and call the tool anyway — that produced a
                    # phantom empty call whose error the model then "retried", inflating the
                    # tool counts (e.g. the double get_task_output). Hand the parse error back
                    # as the tool result so the model re-emits valid JSON instead.
                    obs.log("ARG PARSE ERROR", f"{e}\n{tc.function.arguments}")
                    log_parse_failure(name, tc.function.arguments, str(e))
                    tool_errors_by_name[name] = tool_errors_by_name.get(name, 0) + 1
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": name,
                        "content": json.dumps({
                            "error": f"could not parse tool arguments as JSON ({e}); "
                                     "re-send this call with valid JSON arguments."
                        }),
                    })
                    continue
                # Every save tool is version-stamped here, not by the LLM: the row must
                # key on (eval_id, git_commit) to join its run row and to keep results
                # from different code versions apart.
                if name in SAVE_TOOLS:
                    if eval_id is not None:
                        args["eval_id"] = eval_id        # authoritative id, not the LLM's
                    if git_commit is not None:
                        args["git_commit"] = git_commit  # code_version, not the LLM's
                t_tool = time.perf_counter()
                tool_result = await call_mcp_tool(name, args, verbose=verbose)
                tool_time_by_name[name] = tool_time_by_name.get(name, 0.0) + (time.perf_counter() - t_tool)
                if '"error"' in tool_result or "isError=True" in tool_result:
                    tool_errors_by_name[name] = tool_errors_by_name.get(name, 0) + 1
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": name,
                    "content": tool_result,
                })
        else:
            if verbose:
                obs.log("AGENT STOPPED", f"Hit max_steps={max_steps}")

    wall_time_total = time.perf_counter() - wall_t0   # stop before the post-run scrape
    gpu_end = obs.scrape_vllm_metrics(metrics_url) if metrics_url else None

    return {
        "answer": answer,
        "usage": total_usage,
        "messages": messages,
        "steps": steps_run,
        "stopped_reason": stopped_reason,
        "steps_detail": steps_detail,
        "tool_calls_by_name": tool_calls_by_name,
        "tool_time_by_name": tool_time_by_name,
        "tool_errors_by_name": tool_errors_by_name,
        "llm_time_total": llm_time_total,
        # llm_time_total spans the whole retry sequence; these split out what the model
        # actually spent from what throttling cost us (backoff + failed attempts).
        "llm_time_productive": llm_time_productive_total,
        "llm_retry_wait": llm_retry_wait_total,
        "llm_attempts": attempts_total,
        "wall_time_total": wall_time_total,
        # wall_time minus model latency ≈ tool-execution + agent-loop overhead. The MCP
        # tool server is local for both backends, so this is backend-independent and lets
        # llm_time (model/service latency) be compared without the tool time mixed in.
        "overhead_time": max(0.0, wall_time_total - llm_time_total),
        # ── new derived metrics (plan §2) ──
        "tokens_per_sec": obs.tokens_per_second(total_usage["completion_tokens"], llm_time_total),
        "peak_context": obs.peak_context(steps_detail),
        "client_ram_mb": obs.client_ram_mb(),
        "gpu_start": gpu_start,
        "gpu_end": gpu_end,
        # Weave UI link for THIS run (resolved inside the run_agent op); None when
        # Weave is off. runner.run_one threads it onto the Parquet row.
        "weave_trace_url": obs.current_trace_url(),
    }
