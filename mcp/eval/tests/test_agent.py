"""Tests for eval.agent — retry predicate, session binding, and the loop (mocked)."""

from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from eval import agent


# ── fakes ────────────────────────────────────────────────────────────

class FakeTextItem:
    type = "text"

    def __init__(self, text):
        self.text = text


class FakeMCPResult:
    def __init__(self, text, is_error=False):
        self.content = [FakeTextItem(text)]
        self.isError = is_error


class FakeSession:
    """Records calls; returns scripted JSON text per tool."""

    def __init__(self, responses=None):
        self.responses = responses or {}
        self.calls = []

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        return FakeMCPResult(self.responses.get(name, '{"ok": true}'))


def _msg(content=None, tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls)


def _tool_call(call_id, name, arguments):
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=arguments))


def _response(message, finish_reason, usage):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=finish_reason)],
        usage=SimpleNamespace(model_dump=lambda u=usage: u),
    )


def make_scripted_llm_step(responses):
    """Return an async llm_step that yields each scripted response in order."""
    it = iter(responses)

    async def llm_step(messages):
        return next(it)

    return llm_step


# ── retry predicate ──────────────────────────────────────────────────

def test_is_retryable_by_class_name():
    assert agent._is_retryable(ConnectionResetError("reset"))
    assert agent._is_retryable(TimeoutError("nope"))


def test_is_retryable_by_message():
    assert agent._is_retryable(RuntimeError("the connection reset by peer"))
    assert agent._is_retryable(RuntimeError("operation timed out"))


def test_not_retryable_for_value_error():
    assert not agent._is_retryable(ValueError("bad arg"))


# ── session binding ──────────────────────────────────────────────────

async def test_call_mcp_tool_without_session_raises():
    with pytest.raises(RuntimeError, match="no MCP session"):
        await agent.call_mcp_tool("list_outputs", {}, verbose=False)


async def test_call_mcp_tool_uses_contextvar_session():
    session = FakeSession({"word_iou": '{"word_iou": 0.5}'})
    token = agent._session_var.set(session)
    try:
        text = await agent.call_mcp_tool("word_iou", {"predicted": "a", "expected": "b"}, verbose=False)
    finally:
        agent._session_var.reset(token)
    assert '"word_iou": 0.5' in text
    assert session.calls == [("word_iou", {"predicted": "a", "expected": "b"})]


async def test_make_llm_step_awaits_async_client():
    # Exercises the real make_llm_step: @op + tenacity @retry over an async client.
    class FakeCompletions:
        def __init__(self):
            self.kwargs = None

        async def create(self, **kwargs):
            self.kwargs = kwargs
            return "RESP"

    comps = FakeCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=comps))
    step = agent.make_llm_step(client, "m", tools=[], completion_kwargs={"temperature": 0})
    out = await step([{"role": "user", "content": "hi"}])
    assert out == "RESP"
    assert comps.kwargs["model"] == "m"
    assert comps.kwargs["tool_choice"] == "auto"


async def test_make_llm_step_retries_transient_then_succeeds():
    # tenacity must retry a coroutine function on a retryable error.
    class FlakyCompletions:
        def __init__(self):
            self.n = 0

        async def create(self, **kwargs):
            self.n += 1
            if self.n < 2:
                raise ConnectionResetError("transient")
            return "OK"

    comps = FlakyCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=comps))
    step = agent.make_llm_step(client, "m", tools=[], completion_kwargs={})
    assert await step([]) == "OK"
    assert comps.n == 2


async def test_call_tool_retries_then_succeeds():
    class FlakySession:
        def __init__(self):
            self.n = 0

        async def call_tool(self, name, arguments):
            self.n += 1
            if self.n < 3:
                raise ConnectionResetError("transient")
            return FakeMCPResult('{"recovered": true}')

    session = FlakySession()
    result = await agent._call_tool_retrying(session, "x", {})
    assert session.n == 3
    assert result.isError is False


# ── full loop ────────────────────────────────────────────────────────

async def test_run_agent_happy_path(monkeypatch):
    session = FakeSession({
        "get_task_output": '{"fields": [{"field": "f", "predicted": "3", "expected": "3"}]}',
    })

    @asynccontextmanager
    async def fake_use_session(url):
        tok = agent._session_var.set(session)
        try:
            yield session
        finally:
            agent._session_var.reset(tok)

    monkeypatch.setattr(agent, "use_session", fake_use_session)

    responses = [
        _response(
            _msg(tool_calls=[_tool_call("c1", "get_task_output", '{"task_id": "2450", "run_id": 0}')]),
            "tool_calls",
            {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
        ),
        _response(
            _msg(content="Evaluated field f with word_iou."),
            "stop",
            {"prompt_tokens": 120, "completion_tokens": 20, "total_tokens": 140},
        ),
    ]

    result = await agent.run_agent(
        "Evaluate task 2450", "SYSTEM", make_scripted_llm_step(responses),
        mcp_url="http://fake/mcp", verbose=False,
    )

    assert result["stopped_reason"] == "answered"
    assert result["steps"] == 2
    assert result["usage"] == {"prompt_tokens": 220, "completion_tokens": 30, "total_tokens": 250}
    assert result["peak_context"] == 120
    assert result["tokens_per_sec"] is not None and result["tokens_per_sec"] > 0
    assert result["tool_calls_by_name"] == {"get_task_output": 1}
    assert session.calls[0][0] == "get_task_output"


async def test_run_agent_stops_at_max_steps(monkeypatch):
    session = FakeSession()

    @asynccontextmanager
    async def fake_use_session(url):
        tok = agent._session_var.set(session)
        try:
            yield session
        finally:
            agent._session_var.reset(tok)

    monkeypatch.setattr(agent, "use_session", fake_use_session)

    # Always returns a tool call -> never answers -> hits max_steps.
    async def always_tool(messages):
        return _response(
            _msg(tool_calls=[_tool_call("c", "evaluate_raw_string", "{}")]),
            "tool_calls",
            {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        )

    result = await agent.run_agent(
        "go", "SYS", always_tool, mcp_url="http://fake/mcp", max_steps=3, verbose=False,
    )
    assert result["stopped_reason"] == "max_steps"
    assert result["steps"] == 3


async def test_run_agent_handles_llm_error(monkeypatch):
    session = FakeSession()

    @asynccontextmanager
    async def fake_use_session(url):
        tok = agent._session_var.set(session)
        try:
            yield session
        finally:
            agent._session_var.reset(tok)

    monkeypatch.setattr(agent, "use_session", fake_use_session)

    async def boom(messages):
        raise ValueError("model exploded")

    result = await agent.run_agent(
        "go", "SYS", boom, mcp_url="http://fake/mcp", verbose=False,
    )
    assert result["stopped_reason"] == "error"
    assert "model exploded" in result["answer"]
