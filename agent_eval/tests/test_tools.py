"""Tests for the Mongo write paths — eval_id keying + the agentic_runs mirror.

No real Mongo: get_db is monkeypatched to a fake collection that records calls.
"""

from agent_eval import tools


class _FakeColl:
    def __init__(self):
        self.calls = []

    def replace_one(self, filter_key, doc, upsert=False):
        self.calls.append((filter_key, doc, upsert))

    # _ensure_indexes touches these; make them harmless no-ops.
    def list_indexes(self):
        return iter([])

    def create_index(self, *a, **k):
        return "ix"

    def drop_index(self, name):
        pass


class _FakeDB:
    def __init__(self):
        self.colls: dict = {}

    def __getitem__(self, name):
        return self.colls.setdefault(name, _FakeColl())


def _patch_db(monkeypatch):
    fake = _FakeDB()
    monkeypatch.setattr(tools, "get_db", lambda: fake)
    monkeypatch.setattr(tools, "_indexes_ready", False)
    return fake


def test_save_run_row_keys_on_eval_id_when_present(monkeypatch):
    fake = _patch_db(monkeypatch)
    row = {
        "eval_id": 42, "task_id": "2450", "run_id": 3, "benchmark_id": "5", "model_id": "1",
        "backend": "playground", "agent_model_key": 1, "prompt_key": 1,
        "selection_accuracy": 1.0,
    }
    res = tools.save_run_row(row)
    filter_key, doc, upsert = fake.colls[tools.AGENTIC_RUNS_COLL].calls[0]
    assert filter_key == {"eval_id": 42}          # shared join key wins
    assert upsert is True and "stored_at" in doc
    assert res["collection"] == "agentic_runs"


def test_save_run_row_falls_back_to_composite_without_eval_id(monkeypatch):
    fake = _patch_db(monkeypatch)
    row = {
        "task_id": "2450", "run_id": 3, "benchmark_id": "5", "model_id": "1",
        "backend": "playground", "agent_model_key": 1, "prompt_key": 1,
    }
    tools.save_run_row(row)
    filter_key, _, _ = fake.colls[tools.AGENTIC_RUNS_COLL].calls[0]
    assert filter_key == {k: row[k] for k in tools.RUN_KEY_FIELDS}


def test_save_agentic_evaluation_keys_on_eval_id(monkeypatch):
    fake = _patch_db(monkeypatch)
    fe = [{"field": "x", "field_type": "list", "metric": "evaluate_list", "scores": {}}]
    tools.save_agentic_evaluation(
        task_id="2450", benchmark_id="5", model_id="1", run_id=3,
        image_id=None, field_evaluations=fe, eval_id=42)
    filter_key, doc, upsert = fake.colls[tools.AGENTIC_EVAL_COLL].calls[0]
    assert filter_key == {"eval_id": 42}          # per-judge uniqueness, no collapse
    assert doc["eval_id"] == 42 and doc["task_id"] == "2450"  # identity still stored
    assert upsert is True


def test_save_agentic_evaluation_falls_back_without_eval_id(monkeypatch):
    fake = _patch_db(monkeypatch)
    fe = [{"field": "x", "field_type": "list", "metric": "evaluate_list", "scores": {}}]
    tools.save_agentic_evaluation(
        task_id="2450", benchmark_id="5", model_id="1", run_id=3,
        image_id=None, field_evaluations=fe)
    filter_key, doc, _ = fake.colls[tools.AGENTIC_EVAL_COLL].calls[0]
    assert filter_key == {"task_id": "2450", "benchmark_id": "5", "model_id": "1", "run_id": 3}
    assert doc["eval_id"] is None
