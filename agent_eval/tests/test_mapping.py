"""Tests for eval.mapping — the eval-run registry (build / dedupe / sample / IO)."""

from agent_eval import mapping


OUTPUTS = [
    {"task_id": "2450", "run_id": 0, "benchmark_id": "5", "model_id": "1"},
    {"task_id": "2451", "run_id": None, "benchmark_id": "7", "model_id": "1"},
]
JUDGE = [{"judge_backend": "playground", "judge_model": "gpt-5-mini", "judge_prompt": "composite_v1"}]


def test_build_rows_cross_product():
    rows = mapping.build_rows(OUTPUTS, JUDGE)
    assert len(rows) == 2  # 2 outputs x 1 judge config
    assert rows[0]["task_id"] == "2450" and rows[0]["judge_backend"] == "playground"
    # two judge configs -> doubles
    rows2 = mapping.build_rows(OUTPUTS, JUDGE + [{"judge_backend": "playground",
                                                  "judge_model": "gpt-4o",
                                                  "judge_prompt": "composite_v1"}])
    assert len(rows2) == 4


def test_dedupe_and_assign_ids_and_idempotency():
    cand = mapping.build_rows(OUTPUTS, JUDGE)
    first = mapping.dedupe_and_assign([], cand)
    assert [r["eval_id"] for r in first] == [0, 1]
    # Re-running with the same candidates against the now-existing rows adds nothing.
    assert mapping.dedupe_and_assign(first, cand) == []
    # A genuinely new output continues the id counter.
    more = mapping.build_rows(
        [{"task_id": "2452", "run_id": 0, "benchmark_id": "11", "model_id": "1"}], JUDGE)
    added = mapping.dedupe_and_assign(first, more)
    assert [r["eval_id"] for r in added] == [2]


def test_stratified_sample_is_deterministic_and_spread():
    rows = mapping.build_rows(
        [{"task_id": str(i), "run_id": 0, "benchmark_id": ("5" if i % 2 else "7"), "model_id": "1"}
         for i in range(20)], JUDGE)
    s1 = mapping.stratified_sample(rows, 6, seed=0)
    s2 = mapping.stratified_sample(rows, 6, seed=0)
    assert [r["task_id"] for r in s1] == [r["task_id"] for r in s2]  # deterministic
    assert len(s1) == 6
    assert {r["benchmark_id"] for r in s1} == {"5", "7"}             # both strata present


def test_sample_returns_all_when_n_exceeds_size():
    rows = mapping.build_rows(OUTPUTS, JUDGE)
    assert len(mapping.stratified_sample(rows, 100)) == len(rows)


def test_csv_roundtrip_append_and_read_row(tmp_path):
    path = tmp_path / "eval_mapping.csv"
    rows = mapping.dedupe_and_assign([], mapping.build_rows(OUTPUTS, JUDGE))
    mapping.append_csv(path, rows)
    # append again with a new row -> header not duplicated, both present
    more = mapping.dedupe_and_assign(
        rows, mapping.build_rows(
            [{"task_id": "2452", "run_id": 5, "benchmark_id": "11", "model_id": "1"}], JUDGE))
    mapping.append_csv(path, more)

    back = mapping.read_csv(path)
    assert len(back) == 3
    assert back[0]["eval_id"] == "0"                # CSV values are strings
    row2 = mapping.read_mapping_row(path, 2)
    assert row2["task_id"] == "2452" and row2["run_id"] == "5"


def test_coerce_run_id():
    assert mapping.coerce_run_id("0") == 0
    assert mapping.coerce_run_id("5") == 5
    assert mapping.coerce_run_id("") is None
    assert mapping.coerce_run_id("None") is None
    assert mapping.coerce_run_id(None) is None
