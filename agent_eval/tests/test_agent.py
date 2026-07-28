"""Tests for eval.agent — retry predicate, session binding, and the loop (mocked)."""

from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from agent_eval.runtime import agent


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
    # wall = llm + overhead, and overhead is non-negative
    assert result["overhead_time"] >= 0
    assert abs(result["wall_time_total"] - (result["llm_time_total"] + result["overhead_time"])) < 1e-6
    assert result["tool_calls_by_name"] == {"get_task_output": 1}
    assert session.calls[0][0] == "get_task_output"
    # weave_trace_url is present on the result and null when Weave is disabled (tests)
    assert "weave_trace_url" in result and result["weave_trace_url"] is None


async def test_run_agent_stamps_eval_id_and_git_commit_on_save_evaluation(monkeypatch):
    # The LLM omits eval_id/git_commit; the client injects the authoritative values.
    session = FakeSession({"save_evaluation": '{"saved": true}'})

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
            _msg(tool_calls=[_tool_call("c1", "save_evaluation",
                                        '{"task_id": "2450", "field_evaluations": [{"f": 1}]}')]),
            "tool_calls",
            {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        ),
        _response(_msg(content="done"), "stop",
                  {"prompt_tokens": 12, "completion_tokens": 6, "total_tokens": 18}),
    ]

    await agent.run_agent(
        "Evaluate task 2450", "SYSTEM", make_scripted_llm_step(responses),
        mcp_url="http://fake/mcp", verbose=False, eval_id=77, git_commit="abc123",
    )

    name, args = session.calls[0]
    assert name == "save_evaluation"
    assert args["eval_id"] == 77               # injected client-side, not from the LLM
    assert args["git_commit"] == "abc123"      # code_version stamped client-side too
    assert args["task_id"] == "2450"           # LLM-provided args preserved


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


async def test_run_agent_parse_failure_returns_error_not_empty_call(monkeypatch):
    # Malformed tool-call JSON must NOT dispatch a phantom empty call; instead the model
    # gets an error tool-result it can correct from. (The #3 fix.)
    session = FakeSession()

    @asynccontextmanager
    async def fake_use_session(url):
        tok = agent._session_var.set(session)
        try:
            yield session
        finally:
            agent._session_var.reset(tok)

    monkeypatch.setattr(agent, "use_session", fake_use_session)

    responses = [
        _response(_msg(tool_calls=[_tool_call("c1", "get_task_output", "{bad json")]),
                  "tool_calls", {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}),
        _response(_msg(content="done"), "stop",
                  {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}),
    ]
    result = await agent.run_agent(
        "go", "SYS", make_scripted_llm_step(responses), mcp_url="http://fake/mcp", verbose=False)

    assert session.calls == []                      # no phantom get_task_output({}) hit the server
    tool_msgs = [m for m in result["messages"] if isinstance(m, dict) and m.get("role") == "tool"]
    assert len(tool_msgs) == 1 and '"error"' in tool_msgs[0]["content"]
    assert result["tool_errors_by_name"].get("get_task_output") == 1
