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

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from ..reporting import observability as obs
from ..reporting.observability import op

# The active MCP session for the current asyncio task. asyncio.gather wraps each
# coroutine in its own Task with a copied context, so concurrent run_agent calls
# never see each other's session.
_session_var: contextvars.ContextVar = contextvars.ContextVar("mcp_session", default=None)


# ---------------------------------------------------------------------------
#  Retry policy
# ---------------------------------------------------------------------------

_RETRYABLE_NAMES = {
    "ConnectError", "ConnectionError", "ConnectionResetError", "ConnectTimeout",
    "ReadError", "ReadTimeout", "WriteError", "RemoteProtocolError",
    "TimeoutError", "TimeoutException", "ClosedResourceError", "BrokenResourceError",
    "EndOfStream", "PoolTimeout",
}


def _is_retryable(exc: BaseException) -> bool:
    """True for connection-reset / timeout style errors worth retrying.

    Matches on the exception's class name (so we don't have to import httpx/anyio
    types) and on a "connection"/"timeout"/"reset" substring in the message.
    """
    name = type(exc).__name__
    if name in _RETRYABLE_NAMES:
        return True
    msg = str(exc).lower()
    return any(s in msg for s in ("connection reset", "connection error", "timed out",
                                  "timeout", "broken pipe", "peer closed"))


_retry_policy = dict(
    retry=retry_if_exception(_is_retryable),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=8),
    reraise=True,
)


# ---------------------------------------------------------------------------
#  Session management
# ---------------------------------------------------------------------------

@asynccontextmanager
async def mcp_session(url):
    """Open one initialized MCP ClientSession over streamable HTTP."""
    async with streamablehttp_client(url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


@asynccontextmanager
async def use_session(url):
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


async def load_tools_from_mcp(url):
    """Discover tools from the live server, as OpenAI Chat Completions schemas."""
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
    ]


# ---------------------------------------------------------------------------
#  Tool call + LLM step (both retried)
# ---------------------------------------------------------------------------

@retry(**_retry_policy)
async def _call_tool_retrying(session, tool_name, arguments):
    result = await session.call_tool(tool_name, arguments)
    return result


@op
async def call_mcp_tool(tool_name, arguments, verbose=True):
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
def log_parse_failure(tool_name, raw_args, error_msg):
    """Traced event for JSON argument-parse failures (behavior unchanged)."""
    return {"tool_name": tool_name, "raw_args": raw_args, "error": error_msg}


def make_llm_step(client, model, tools, completion_kwargs):
    """Build a retried, traced async llm_step bound to a client/model/tools/kwargs.

    Returned as a closure so run_agent stays backend-agnostic and tests can pass
    a fake client. `llm_step` is a COROUTINE: run_agent awaits it, so concurrent
    runs (asyncio.gather in runner.run_batch) truly overlap at the LLM layer
    instead of serializing on a blocking call. Retry covers transient connection
    resets (tenacity supports coroutine functions); @op traces async calls too.
    """

    @op
    @retry(**_retry_policy)
    async def llm_step(messages):
        return await client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            **completion_kwargs,
        )

    return llm_step


# ---------------------------------------------------------------------------
#  Agent loop
# ---------------------------------------------------------------------------

@op
async def run_agent(
    user_prompt,
    system_prompt,
    llm_step,
    mcp_url,
    max_steps=12,
    verbose=True,
    backend=None,
    model=None,
    task_id=None,
    run_id=None,
    metrics_url=None,
):
    """Run the agent on one output. Opens one MCP session for the whole loop.

    `llm_step` is the closure from make_llm_step. `metrics_url`, when given, is
    scraped once before and once after the run for GPU/queue context.
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
    wall_t0 = time.perf_counter()
    steps_run = 0
    stopped_reason = "max_steps"
    answer = "Stopped after maximum tool-calling steps."

    gpu_start = obs.scrape_vllm_metrics(metrics_url) if metrics_url else None

    if verbose:
        obs.log("AGENT START", f"Model: {model}  Backend: {backend}")

    async with use_session(mcp_url):
        for step in range(1, max_steps + 1):
            steps_run = step
            if verbose:
                obs.log(f"LLM CALL {step}", f"{len(messages)} messages")

            t0 = time.perf_counter()
            try:
                response = await llm_step(messages)
            except Exception as e:  # noqa: BLE001
                stopped_reason = "error"
                answer = f"LLM error: {e}"
                if verbose:
                    obs.log("LLM ERROR", str(e))
                break
            dt = time.perf_counter() - t0
            llm_time_total += dt

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
                "llm_time": dt,
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
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError as e:
                    obs.log("ARG PARSE ERROR", f"{e}\n{tc.function.arguments}")
                    # BUG (logged, not fixed): args fall back to {} — tool still called.
                    log_parse_failure(name, tc.function.arguments, str(e))
                    args = {}
                tool_calls_by_name[name] = tool_calls_by_name.get(name, 0) + 1
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

    gpu_end = obs.scrape_vllm_metrics(metrics_url) if metrics_url else None
    wall_time_total = time.perf_counter() - wall_t0

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
        "wall_time_total": wall_time_total,
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
