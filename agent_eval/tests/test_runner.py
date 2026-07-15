"""Tests for eval.runner — row discovery and bounded-parallel batch."""

import asyncio
from contextlib import asynccontextmanager

from agent_eval.runtime import runner


async def test_discover_rows_parses_listings(monkeypatch):
    @asynccontextmanager
    async def fake_use_session(url):
        yield None

    listings = {
        "5": '{"outputs": [{"task_id": "2450", "run_id": 0, "benchmark_id": "5", "model_id": "1"}]}',
        "6": '{"outputs": [{"task_id": "3780", "run_id": null, "benchmark_id": "6", "model_id": "1"}]}',
    }

    async def fake_call(tool_name, arguments, verbose=True):
        return listings[arguments["benchmark_id"]]

    monkeypatch.setattr(runner, "use_session", fake_use_session)
    monkeypatch.setattr(runner, "call_mcp_tool", fake_call)

    rows = await runner.discover_rows("http://x/mcp", ["5", "6"], limit=5)
    assert len(rows) == 2
    assert rows[0]["task_id"] == "2450"
    assert rows[1]["run_id"] is None  # benchmark 6 has no run_id


async def test_run_batch_respects_concurrency(monkeypatch):
    monkeypatch.setattr(runner, "_load_gold_metrics", lambda: {})

    active = 0
    peak = 0

    async def fake_run_one(row, ctx, gold, concurrency, max_steps, verbose,
                           write_sink, gpu_metrics, write_mongo=True):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        return {"task_id": row["task_id"], "stopped_reason": "answered"}

    monkeypatch.setattr(runner, "run_one", fake_run_one)

    rows = [{"task_id": str(i), "run_id": 0, "benchmark_id": "5"} for i in range(6)]
    results = await runner.run_batch(rows, ctx={}, concurrency=2, write_sink=False,
                                     gpu_metrics=False, write_mongo=False)

    assert len(results) == 6
    assert peak <= 2  # semaphore honored
