"""Tests for the Mongo write paths — version-aware (eval_id, git_commit) keys, the
agentic_runs mirror, and skip-if-exists. No real Mongo: get_db is monkeypatched to a
fake collection that records calls and emulates upsert/count."""

from agent_eval import tools


class _FakeColl:
    def __init__(self):
        self.calls = []
        self.docs: list = []

    @staticmethod
    def _match(doc, flt):
        for k, v in flt.items():
            dv = doc.get(k)
            if isinstance(v, dict) and "$ne" in v:   # emulate Mongo's {$ne: x}
                if dv == v["$ne"]:
                    return False
            elif dv != v:
                return False
        return True

    def replace_one(self, filter_key, doc, upsert=False):
        self.calls.append((filter_key, doc, upsert))
        self.docs = [d for d in self.docs if not self._match(d, filter_key)]
        self.docs.append(dict(doc))

    def count_documents(self, flt, limit=None):
        return sum(1 for d in self.docs if self._match(d, flt))

    # _ensure_indexes touches these; harmless no-ops.
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


def test_save_run_row_keys_on_eval_id_and_version(monkeypatch):
    fake = _patch_db(monkeypatch)
    row = {"eval_id": 42, "git_commit": "abc123", "task_id": "2450", "run_id": 3,
           "benchmark_id": "5", "model_id": "1", "backend": "playground",
           "agent_model_key": 1, "prompt_key": 1, "selection_accuracy": 1.0}
    res = tools.save_run_row(row)
    filter_key, doc, upsert = fake.colls[tools.AGENTIC_RUNS_COLL].calls[0]
    assert filter_key == {"eval_id": 42, "git_commit": "abc123"}   # version-scoped
    assert upsert is True and "stored_at" in doc
    assert res["collection"] == "agentic_runs"


def test_save_run_row_falls_back_to_composite_plus_version(monkeypatch):
    fake = _patch_db(monkeypatch)
    row = {"git_commit": "abc123", "task_id": "2450", "run_id": 3, "benchmark_id": "5",
           "model_id": "1", "backend": "playground", "agent_model_key": 1, "prompt_key": 1}
    tools.save_run_row(row)
    filter_key, _, _ = fake.colls[tools.AGENTIC_RUNS_COLL].calls[0]
    assert filter_key == {**{k: row[k] for k in tools.RUN_KEY_FIELDS}, "git_commit": "abc123"}


def test_save_agentic_evaluation_keys_on_eval_id_and_version(monkeypatch):
    fake = _patch_db(monkeypatch)
    fe = [{"field": "x", "field_type": "list", "metric": "evaluate_list", "scores": {}}]
    tools.save_agentic_evaluation(task_id="2450", benchmark_id="5", model_id="1", run_id=3,
                                  image_id=None, field_evaluations=fe, eval_id=42, git_commit="abc123")
    filter_key, doc, upsert = fake.colls[tools.AGENTIC_EVAL_COLL].calls[0]
    assert filter_key == {"eval_id": 42, "git_commit": "abc123"}
    assert doc["eval_id"] == 42 and doc["git_commit"] == "abc123" and doc["task_id"] == "2450"
    assert upsert is True


def test_save_agentic_evaluation_falls_back_without_eval_id(monkeypatch):
    fake = _patch_db(monkeypatch)
    fe = [{"field": "x", "field_type": "list", "metric": "evaluate_list", "scores": {}}]
    tools.save_agentic_evaluation(task_id="2450", benchmark_id="5", model_id="1", run_id=3,
                                  image_id=None, field_evaluations=fe, git_commit="abc123")
    filter_key, doc, _ = fake.colls[tools.AGENTIC_EVAL_COLL].calls[0]
    assert filter_key == {"task_id": "2450", "benchmark_id": "5", "model_id": "1",
                          "run_id": 3, "git_commit": "abc123"}
    assert doc["eval_id"] is None


def test_run_exists_is_version_scoped(monkeypatch):
    fake = _patch_db(monkeypatch)
    row = {"eval_id": 42, "git_commit": "abc123", "task_id": "2450", "run_id": 3,
           "benchmark_id": "5", "model_id": "1", "backend": "playground",
           "agent_model_key": 1, "prompt_key": 1}
    assert tools.run_exists(row) is False          # nothing saved yet
    tools.save_run_row(row)
    assert tools.run_exists(row) is True           # same identity + version -> exists
    # a NEW code version of the same eval does NOT exist yet (coexists, not overwritten)
    assert tools.run_exists({**row, "git_commit": "def456"}) is False


def test_run_exists_ignores_errored_runs(monkeypatch):
    fake = _patch_db(monkeypatch)
    row = {"eval_id": 42, "git_commit": "abc123", "task_id": "2450", "run_id": 3,
           "benchmark_id": "5", "model_id": "1", "backend": "playground",
           "agent_model_key": 1, "prompt_key": 1}
    tools.save_run_row({**row, "stopped_reason": "error"})    # a prior errored run
    assert tools.run_exists(row) is False                     # errored -> re-runnable, not "exists"
    tools.save_run_row({**row, "stopped_reason": "answered"})  # now a successful run
    assert tools.run_exists(row) is True


def test_versions_coexist_not_overwritten(monkeypatch):
    fake = _patch_db(monkeypatch)
    base = {"eval_id": 42, "task_id": "2450", "run_id": 3, "benchmark_id": "5", "model_id": "1",
            "backend": "playground", "agent_model_key": 1, "prompt_key": 1}
    tools.save_run_row({**base, "git_commit": "v1", "selection_accuracy": 0.0})
    tools.save_run_row({**base, "git_commit": "v2", "selection_accuracy": 1.0})
    coll = fake.colls[tools.AGENTIC_RUNS_COLL]
    assert len(coll.docs) == 2                      # both versions kept
    assert {d["git_commit"] for d in coll.docs} == {"v1", "v2"}
