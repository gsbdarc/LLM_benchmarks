"""Tests for analysis.export_runs — the agentic_runs -> Parquet cache dump (no real Mongo)."""

import pandas as pd

from agent_eval import tools
from analysis import export_runs


class _FakeColl:
    def __init__(self, docs):
        self._docs = docs

    def find(self, flt=None, projection=None):
        # emulate the {"_id": 0} projection export_runs requests
        return iter([{k: v for k, v in d.items() if k != "_id"} for d in self._docs])


class _FakeDB:
    def __init__(self, docs):
        self._docs = docs

    def __getitem__(self, name):
        return _FakeColl(self._docs)


def test_export_runs_writes_parquet_dropping_mongo_fields(tmp_path, monkeypatch):
    docs = [
        {"_id": "x1", "eval_id": 1, "model": "gpt-5-mini", "total_dollar_cost": 0.007, "stored_at": "t"},
        {"_id": "x2", "eval_id": 2, "model": "claude-sonnet-4-6", "total_dollar_cost": 0.075, "stored_at": "t"},
    ]
    monkeypatch.setattr(tools, "get_db", lambda: _FakeDB(docs))

    path = export_runs.export(tmp_path / "cache")

    assert path.parent.name == "data"           # under base/data/ so build_dashboard's glob finds it
    df = pd.read_parquet(path)
    assert len(df) == 2
    assert "_id" not in df.columns and "stored_at" not in df.columns   # Mongo-only fields dropped
    assert set(df["model"]) == {"gpt-5-mini", "claude-sonnet-4-6"}
    assert df.loc[df["model"] == "gpt-5-mini", "total_dollar_cost"].iloc[0] == 0.007
