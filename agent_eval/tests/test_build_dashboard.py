"""Tests for analysis.build_dashboard — snapshot + HTML injection."""

import json

import pytest

from analysis import build_dashboard, queries
from agent_eval import config
from agent_eval.reporting import sink
from agent_eval.tests.test_sink import sample_meta, sample_result


def _write_runs(base_dir):
    rows = [
        sink.flatten_run(sample_result(tokens_per_sec=10.0),
                         sample_meta(backend="nim", model="gemma", prompt_name="baseline_v1", concurrency=1),
                         integrity={"save_success": True, "save_count": 1,
                                    "score_consistency": {"consistent": True}},
                         scores={"selection_accuracy": {"selection_accuracy": 1.0}}),
        sink.flatten_run(sample_result(tokens_per_sec=20.0),
                         sample_meta(backend="nim", model="gemma", prompt_name="variant_v2", concurrency=4),
                         integrity={"save_success": True, "save_count": 1,
                                    "score_consistency": {"consistent": True}},
                         scores={"selection_accuracy": {"selection_accuracy": 0.5}}),
    ]
    sink.write_runs(rows, base_dir=base_dir)


# Existing tests pass summarize=False so they never touch the network.
def test_build_snapshot_counts_runs(tmp_path):
    _write_runs(tmp_path)
    snap = build_dashboard.build_snapshot(tmp_path, summarize=False)
    assert snap["n_runs"] == 2
    assert {r["prompt_name"] for r in snap["runs"]} == {"baseline_v1", "variant_v2"}
    assert "generated_at" in snap


def test_build_snapshot_raises_without_runs(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_dashboard.build_snapshot(tmp_path, summarize=False)


def test_build_snapshot_embeds_prompts_and_glossary(tmp_path):
    from agent_eval.prompts import PROMPT_NAME
    rows = [sink.flatten_run(sample_result(), sample_meta(prompt_name=PROMPT_NAME))]
    sink.write_runs(rows, base_dir=tmp_path)
    snap = build_dashboard.build_snapshot(tmp_path, summarize=False)
    # glossary always embedded
    assert any(g["term"] == "save_success" for g in snap["glossary"])
    # prompt registry carries the current prompt's real text
    assert PROMPT_NAME in snap["prompts"]
    assert snap["prompts"][PROMPT_NAME]["system"]
    assert snap["prompts"][PROMPT_NAME]["user"]
    # no summarizer requested -> no blurbs, no network
    assert snap["path_summaries"] == {}


def test_render_html_injects_and_escapes(tmp_path):
    _write_runs(tmp_path)
    snap = build_dashboard.build_snapshot(tmp_path, summarize=False)
    html = build_dashboard.render_html(snap)
    # placeholder fully replaced
    assert build_dashboard.PLACEHOLDER not in html
    # raw "</script>" must not appear inside the injected data (escaped to <)
    head = html.split('id="agentviz-data"')[1].split("</script>")[0]
    assert "<" not in head.split(">", 1)[1]  # no raw '<' in the JSON payload


def test_render_html_payload_parses(tmp_path):
    _write_runs(tmp_path)
    snap = build_dashboard.build_snapshot(tmp_path, summarize=False)
    html = build_dashboard.render_html(snap)
    # extract the JSON between the data-script tags and confirm it parses back
    payload = html.split('type="application/json">')[1].split("</script>")[0]
    payload = payload.replace("\\u003c", "<")
    data = json.loads(payload)
    assert data["n_runs"] == 2
    assert len(data["runs"]) == 2


def test_main_writes_file(tmp_path):
    _write_runs(tmp_path)
    out = tmp_path / "agent_dashboard.html"
    build_dashboard.main(["--base-dir", str(tmp_path), "--out", str(out), "--no-summaries"])
    assert out.exists()
    assert "Agentic Metric-Eval Dashboard" in out.read_text()


class _FakeCompletions:
    def __init__(self, text):
        self._text = text
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        msg = type("M", (), {"content": self._text})()
        choice = type("C", (), {"message": msg})()
        return type("R", (), {"choices": [choice]})()


def _fake_client(text):
    comps = _FakeCompletions(text)
    client = type("Client", (), {"chat": type("Chat", (), {"completions": comps})()})()
    return client, comps


def _write_pathful_runs(base_dir, n=3):
    result = sample_result(steps_detail=[
        {"step": 1, "thinking_nim": "these are a list of channels", "tool_calls": ["get_task_output"]},
        {"step": 2, "thinking_nim": "routing to the list tool", "tool_calls": ["evaluate_list"]},
        {"step": 3, "tool_calls": ["save_evaluation"]},
    ])
    rows = [sink.flatten_run(result, sample_meta(task_id=str(2450 + i))) for i in range(n)]
    sink.write_runs(rows, base_dir=base_dir)


def test_summarize_paths_generates_and_caches(tmp_path, monkeypatch):
    _write_pathful_runs(tmp_path)
    con = queries.connect(tmp_path)
    present = build_dashboard._present_columns(con)
    client, comps = _fake_client("Routed to the list tool because the values are channel lists.")
    monkeypatch.setattr(config, "sync_openai_client", lambda *a, **k: (client, "m", {}))

    cache = tmp_path / "cache.json"
    sig = "get_task_output → evaluate_list → save_evaluation"
    summaries = build_dashboard.summarize_paths(con, present, "summarizer", cache_path=cache)
    assert sig in summaries and "list" in summaries[sig].lower()
    assert comps.calls == 1
    assert cache.exists()

    # Second call reuses the cache — no new LLM calls.
    summaries2 = build_dashboard.summarize_paths(con, present, "summarizer", cache_path=cache)
    assert summaries2[sig] == summaries[sig]
    assert comps.calls == 1  # unchanged


def test_summarize_paths_graceful_when_summarizer_unbuildable(tmp_path, monkeypatch):
    _write_pathful_runs(tmp_path)
    con = queries.connect(tmp_path)
    present = build_dashboard._present_columns(con)

    def _boom(*a, **k):
        raise ValueError("no such backend")

    monkeypatch.setattr(config, "sync_openai_client", _boom)
    # No cache, summarizer can't be built -> empty dict, no raise.
    assert build_dashboard.summarize_paths(con, present, "nope", cache_path=tmp_path / "c.json") == {}
