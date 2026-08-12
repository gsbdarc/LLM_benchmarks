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


def test_is_retryable_for_transient_api_errors():
    # Matched by class name (the OpenAI SDK's typed transient errors) — no import needed.
    for cls_name in ("RateLimitError", "InternalServerError", "APITimeoutError",
                     "APIConnectionError"):
        exc = type(cls_name, (Exception,), {})("boom")
        assert agent._is_retryable(exc), cls_name
    # And by message signal.
    assert agent._is_retryable(RuntimeError("Error code: 429 - rate limit exceeded"))
    assert agent._is_retryable(RuntimeError("503 service unavailable"))


def test_not_retryable_for_permanent_api_errors():
    # Permanent config/misuse errors must fail fast, not burn retries.
    for cls_name in ("BadRequestError", "AuthenticationError", "NotFoundError"):
        exc = type(cls_name, (Exception,), {})("Error code: 400")
        assert not agent._is_retryable(exc), cls_name


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


# ── tool-list filtering: a run must see only its own task's tools ────────────

class _FakeTool:
    def __init__(self, name):
        self.name = name
        self.description = f"does {name}"
        self.inputSchema = {"type": "object", "properties": {}}


SERVER_TOOLS = [
    "list_outputs", "get_task_output", "save_evaluation",
    "evaluate_raw_string", "evaluate_extracted_string", "evaluate_list",
    "get_guide_date_case", "compute_guide_date", "save_correction",
]


def _fake_mcp_session(monkeypatch):
    from contextlib import asynccontextmanager

    class Session:
        async def list_tools(self):
            return SimpleNamespace(tools=[_FakeTool(n) for n in SERVER_TOOLS])

    @asynccontextmanager
    async def fake(url):
        yield Session()

    monkeypatch.setattr(agent, "mcp_session", fake)


async def test_load_tools_unfiltered_shows_everything(monkeypatch):
    _fake_mcp_session(monkeypatch)
    tools = await agent.load_tools_from_mcp("http://unused/mcp")
    assert {t["function"]["name"] for t in tools} == set(SERVER_TOOLS)
    assert tools[0]["function"]["description"] and tools[0]["function"]["parameters"]


async def test_date_fix_run_cannot_see_list_outputs(monkeypatch):
    """The measured hazard: list_outputs exists for work-list discovery, but its
    description tells the model to use it first, and runs that called it fared worse."""
    _fake_mcp_session(monkeypatch)
    tools = await agent.load_tools_from_mcp(
        "http://unused/mcp", allow=agent.tools_for_prompt("date_fix_v1"))
    names = {t["function"]["name"] for t in tools}
    assert names == {"get_guide_date_case", "compute_guide_date", "save_correction"}
    assert "list_outputs" not in names


async def test_metric_eval_run_sees_its_five_tools(monkeypatch):
    _fake_mcp_session(monkeypatch)
    tools = await agent.load_tools_from_mcp(
        "http://unused/mcp", allow=agent.tools_for_prompt("composite_v2"))
    names = {t["function"]["name"] for t in tools}
    assert names == {"get_task_output", "save_evaluation", "evaluate_raw_string",
                     "evaluate_extracted_string", "evaluate_list"}
    assert "list_outputs" not in names and "compute_guide_date" not in names


def test_tools_for_prompt_defaults_to_metric_eval():
    assert agent.tools_for_prompt(None) is agent.METRIC_EVAL_TOOLS
    assert agent.tools_for_prompt("something_new") is agent.METRIC_EVAL_TOOLS
    assert agent.tools_for_prompt("date_fix_v1") is agent.DATE_FIX_TOOLS


# ── LLM time: productive latency vs retry waste ──────────────────────────────

def test_summarize_attempts_splits_productive_from_retry_wait():
    tally = [{"seconds": 0.1, "ok": False, "error": "APIConnectionError"},
             {"seconds": 0.1, "ok": False, "error": "RateLimitError"},
             {"seconds": 2.0, "ok": True}]
    got = agent._summarize_attempts(tally, step_total=12.0)   # 10s of it was backoff
    assert got["llm_time_productive"] == 2.0
    assert got["llm_retry_wait"] == 10.0
    assert got["attempts"] == 3
    assert got["retry_errors"] == "APIConnectionError,RateLimitError"


def test_summarize_attempts_without_a_tally_keeps_old_behaviour():
    """Faked clients (and older code paths) record nothing; the step total stands in
    so numbers stay comparable rather than collapsing to zero."""
    got = agent._summarize_attempts([], step_total=3.0)
    assert got == {"llm_time_productive": 3.0, "llm_retry_wait": 0.0,
                   "attempts": 1, "retry_errors": None}


def test_retry_after_header_is_honoured_over_short_backoff():
    def state(headers, attempt=1):
        exc = SimpleNamespace(response=SimpleNamespace(headers=headers))
        return SimpleNamespace(attempt_number=attempt, idle_for=0, seconds_since_start=0,
                               retry_object=None, outcome=SimpleNamespace(exception=lambda: exc))
    assert agent._wait_honoring_retry_after(state({"retry-after": "45"})) == 45.0
    # capped, so an absurd header cannot stall a batch
    assert agent._wait_honoring_retry_after(state({"Retry-After": "9999"})) == agent._MAX_RETRY_AFTER
    # absent/garbage -> plain exponential backoff
    assert agent._wait_honoring_retry_after(state({})) == 0.5
    assert agent._wait_honoring_retry_after(state({"retry-after": "soon"})) == 0.5
