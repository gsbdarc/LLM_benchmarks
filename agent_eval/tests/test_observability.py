"""Tests for eval.observability — derivations, hashes, Prometheus parsing, scrape."""

import json

from agent_eval.reporting import observability as obs


# ── hashes ───────────────────────────────────────────────────────────

def test_prompt_hash_stable_and_8_chars():
    h = obs.compute_prompt_hash("hello world")
    assert len(h) == 8
    assert h == obs.compute_prompt_hash("hello world")


def test_prompt_hash_changes_with_text():
    assert obs.compute_prompt_hash("a") != obs.compute_prompt_hash("b")


def test_tools_hash_order_independent_for_dict_keys():
    a = [{"name": "x", "p": 1}]
    b = [{"p": 1, "name": "x"}]
    assert obs.compute_tools_hash(a) == obs.compute_tools_hash(b)


# ── thinking / reasoning extraction ──────────────────────────────────

def test_split_thinking_extracts_block():
    reasoning, visible = obs.split_thinking("<think>plan here</think>answer")
    assert reasoning == "plan here"
    assert visible == "answer"


def test_split_thinking_no_block():
    reasoning, visible = obs.split_thinking("just an answer")
    assert reasoning == ""
    assert visible == "just an answer"


def test_split_thinking_none():
    assert obs.split_thinking(None) == ("", "")


# ── tokens/sec + peak context ────────────────────────────────────────

def test_tokens_per_second_basic():
    assert obs.tokens_per_second(100, 4.0) == 25.0


def test_tokens_per_second_zero_time_is_none():
    assert obs.tokens_per_second(100, 0) is None


def test_tokens_per_second_none_tokens():
    assert obs.tokens_per_second(None, 5.0) is None


def test_peak_context_picks_max_prompt_tokens():
    steps = [
        {"usage": {"prompt_tokens": 100}},
        {"usage": {"prompt_tokens": 350}},
        {"usage": {"prompt_tokens": 200}},
    ]
    assert obs.peak_context(steps) == 350


def test_peak_context_empty_is_none():
    assert obs.peak_context([]) is None
    assert obs.peak_context([{"usage": None}]) is None


def test_tool_sequence_flattens_in_order():
    steps = [
        {"tool_calls": ["get_task_output"]},
        {"tool_calls": ["null_accuracy", "levenshtein"]},
        {"tool_calls": ["save_evaluation"]},
        {"tool_calls": []},
    ]
    assert obs.tool_sequence(steps) == [
        "get_task_output", "null_accuracy", "levenshtein", "save_evaluation",
    ]


def test_tool_sequence_empty():
    assert obs.tool_sequence([]) == []
    assert obs.tool_sequence(None) == []


def test_reasoning_blob_collects_thinking_steps():
    import json
    steps = [
        {"step": 1, "thinking_nim": "a list of channels", "tool_calls": ["evaluate_list"]},
        {"step": 2, "thinking_nim": None, "thinking_api": None, "tool_calls": ["save_evaluation"]},
        {"step": 3, "thinking_api": "double-checking", "tool_calls": []},
    ]
    blob = json.loads(obs.reasoning_blob(steps))
    assert [b["step"] for b in blob] == [1, 3]  # step 2 had no reasoning
    assert blob[0]["tool_calls"] == ["evaluate_list"]


def test_reasoning_blob_truncates_to_budget():
    import json
    steps = [{"step": 1, "thinking_nim": "x" * 5000, "tool_calls": []}]
    blob = json.loads(obs.reasoning_blob(steps, max_chars=100))
    assert len(blob[0]["reasoning"]) <= 101  # 100 + the ellipsis


def test_reasoning_blob_none_when_no_reasoning():
    assert obs.reasoning_blob([{"step": 1, "tool_calls": ["get_task_output"]}]) is None
    assert obs.reasoning_blob([]) is None
    assert obs.reasoning_blob(None) is None


def test_client_ram_mb_returns_positive():
    ram = obs.client_ram_mb()
    assert ram is None or ram > 0


def test_current_trace_url_none_when_weave_disabled():
    # conftest sets EVAL_DISABLE_WEAVE=1, so there's no active Weave call.
    assert obs.current_trace_url() is None


# ── Prometheus parsing ───────────────────────────────────────────────

SAMPLE_METRICS = """\
# HELP vllm:gpu_cache_usage_perc GPU KV-cache usage.
# TYPE vllm:gpu_cache_usage_perc gauge
vllm:gpu_cache_usage_perc{model_name="gemma"} 0.42
vllm:num_requests_running{model_name="gemma"} 2.0
vllm:num_requests_waiting{model_name="gemma"} 1.0
vllm:prompt_tokens_total{model_name="gemma"} 12345.0
vllm:generation_tokens_total{model_name="gemma"} 6789.0
"""


def test_parse_prometheus_strips_labels():
    parsed = obs.parse_prometheus(SAMPLE_METRICS)
    assert parsed["vllm:gpu_cache_usage_perc"] == 0.42
    assert parsed["vllm:num_requests_running"] == 2.0


def test_parse_prometheus_skips_comments_and_blanks():
    parsed = obs.parse_prometheus("# comment\n\nvllm:x 1.0\n")
    assert parsed == {"vllm:x": 1.0}


def test_parse_prometheus_sums_label_sets():
    text = 'foo{a="1"} 2.0\nfoo{a="2"} 3.0\n'
    assert obs.parse_prometheus(text)["foo"] == 5.0


def test_select_vllm_metrics_maps_keys():
    selected = obs.select_vllm_metrics(obs.parse_prometheus(SAMPLE_METRICS))
    assert selected["gpu_cache_usage_perc"] == 0.42
    assert selected["num_requests_running"] == 2.0
    assert selected["num_requests_waiting"] == 1.0
    assert selected["prompt_tokens_total"] == 12345.0
    assert selected["generation_tokens_total"] == 6789.0


def test_select_vllm_metrics_handles_unprefixed():
    parsed = {"gpu_cache_usage_perc": 0.1}
    assert obs.select_vllm_metrics(parsed)["gpu_cache_usage_perc"] == 0.1


def test_select_vllm_metrics_missing_is_none():
    assert obs.select_vllm_metrics({})["num_requests_running"] is None


# ── scrape (no real network) ─────────────────────────────────────────

def test_scrape_returns_error_dict_on_failure():
    # Unroutable port -> connection error -> {"error": ...}, never raises.
    result = obs.scrape_vllm_metrics("http://127.0.0.1:1/metrics", timeout=0.2)
    assert "error" in result


def test_scrape_parses_mocked_response(monkeypatch):
    class FakeResp:
        text = SAMPLE_METRICS

        def raise_for_status(self):
            pass

    import requests

    monkeypatch.setattr(requests, "get", lambda *a, **k: FakeResp())
    result = obs.scrape_vllm_metrics("http://fake/metrics")
    assert result["gpu_cache_usage_perc"] == 0.42
    assert "error" not in result


# ── steps_trace: the per-step record error analysis reads ────────────────────

def _step(n, visible="thinking out loud", args='{"city": "Palo Alto"}'):
    return {
        "step": n, "finish_reason": "tool_calls", "visible": visible,
        "tool_calls": ["get_weather"],
        "tool_call_args": [{"name": "get_weather", "arguments": args}],
        "llm_time": 1.5, "llm_time_productive": 1.0, "attempts": 2,
        "usage": {"prompt_tokens": 100, "completion_tokens": 20,
                  "completion_tokens_details": {"reasoning_tokens": 64}},
    }


def test_steps_trace_keeps_order_prose_and_arguments():
    trace = json.loads(obs.steps_trace([_step(1), _step(2)]))
    assert [e["step"] for e in trace] == [1, 2]
    assert trace[0]["visible"] == "thinking out loud"
    # The ARGUMENTS are the diagnostic part — a wrong value passed to a tool is the bug.
    assert trace[0]["tool_calls"] == [{"name": "get_weather", "arguments": '{"city": "Palo Alto"}'}]
    assert trace[0]["prompt_tokens"] == 100 and trace[0]["attempts"] == 2


def test_steps_trace_preserves_malformed_tool_json_verbatim():
    """Raw argument strings, so a JSON failure stays visible instead of normalised away."""
    trace = json.loads(obs.steps_trace([_step(1, args='{"city": "Palo A')]))
    assert trace[0]["tool_calls"][0]["arguments"] == '{"city": "Palo A'


def test_steps_trace_truncates_long_prose_and_marks_it():
    trace = json.loads(obs.steps_trace([_step(1, visible="x" * 5000)], max_chars=100))
    assert len(trace[0]["visible"]) == 101          # 100 chars + the ellipsis
    assert trace[0]["truncated"] is True


def test_steps_trace_bounds_step_count_and_says_so():
    trace = json.loads(obs.steps_trace([_step(i) for i in range(1, 40)], max_steps=5))
    assert len(trace) == 6                          # 5 steps + the omission note
    assert "34 further steps omitted" in trace[-1]["note"]


def test_steps_trace_none_when_no_steps():
    assert obs.steps_trace(None) is None
    assert obs.steps_trace([]) is None


def test_reasoning_tokens_sums_when_reported_else_none():
    assert obs.reasoning_tokens([_step(1), _step(2)]) == 128
    assert obs.reasoning_tokens([{"usage": {"prompt_tokens": 5}}]) is None
    assert obs.reasoning_tokens([]) is None
