"""Tests for eval.mapping — the eval-run registry (build / dedupe / sample / IO)."""

from agent_eval.registry import mapping


OUTPUTS = [
    {"task_id": "2450", "run_id": 0, "benchmark_id": "5", "model_id": "1"},
    {"task_id": "2451", "run_id": None, "benchmark_id": "7", "model_id": "1"},
]
JUDGE = [{"judge_backend": "playground", "judge_model": "gpt-5-mini", "judge_prompt": "composite_v1"}]


THREE_JUDGES = [
    {"judge_backend": "playground", "judge_model": m, "judge_prompt": "composite_v1"}
    for m in ("gpt-5-mini", "DeepSeek-V3.2", "claude-sonnet-4-6")
]


def test_build_rows_cross_product():
    rows = mapping.build_rows(OUTPUTS, JUDGE)
    assert len(rows) == 2  # 2 outputs x 1 judge config
    assert rows[0]["task_id"] == "2450" and rows[0]["judge_backend"] == "playground"


def test_sample_paired_keeps_all_judges_for_sampled_outputs():
    # 6 outputs across 2 benchmarks, each crossed with 3 judges = 18 rows.
    outs = [{"task_id": str(i), "run_id": 0,
             "benchmark_id": ("5" if i % 2 else "7"), "model_id": "1"} for i in range(6)]
    rows = mapping.build_rows(outs, THREE_JUDGES)
    assert len(rows) == 18

    picked = mapping.sample_paired(rows, k_outputs=4, seed=0)
    # 4 outputs x 3 judges = 12 rows, and every sampled output keeps all 3 judges.
    assert len(picked) == 12
    by_output: dict = {}
    for r in picked:
        by_output.setdefault(mapping._output_key(r), set()).add(r["judge_model"])
    assert len(by_output) == 4
    assert all(js == {"gpt-5-mini", "DeepSeek-V3.2", "claude-sonnet-4-6"} for js in by_output.values())
    # deterministic
    assert mapping.sample_paired(rows, 4, seed=0) == picked


def test_resolve_prompt_names_and_judge_configs():
    from agent_eval.registry import create_eval_mapping as cem
    # default = every registered variant (v1 + v2 both ship in the repo)
    alln = cem.resolve_prompt_names(None)
    assert "composite_v1" in alln and "composite_v2" in alln
    # names and indices both resolve; order preserved
    assert cem.resolve_prompt_names("2,composite_v1") == ["composite_v2", "composite_v1"]
    # judge_configs crosses models x prompts (one config per model for a single prompt)
    cfgs = cem.judge_configs(["composite_v2"])
    assert cfgs and all(c["judge_prompt"] == "composite_v2" for c in cfgs)
    assert len({c["judge_model"] for c in cfgs}) == len(cfgs)


def test_select_by_outputs_reuses_exact_outputs():
    # A registry with two prompts per output; reuse a v1 reference's outputs for v2.
    outs = [{"task_id": str(i), "run_id": 0, "benchmark_id": "5", "model_id": "1"} for i in range(5)]
    two_prompts = [
        {"judge_backend": "playground", "judge_model": "gpt-5-mini", "judge_prompt": p}
        for p in ("composite_v1", "composite_v2")
    ]
    full = mapping.build_rows(outs, two_prompts)  # 5 outputs x 2 prompts = 10 rows
    reference = [r for r in mapping.build_rows(outs[:3], two_prompts)
                 if r["judge_prompt"] == "composite_v1"]  # v1 sample over outputs 0,1,2

    picked = mapping.select_by_outputs(full, reference)
    assert {mapping._output_key(r) for r in picked} == {mapping._output_key(r) for r in reference}
    # filtering to just v2 gives one row per reference output
    v2 = [r for r in picked if r["judge_prompt"] == "composite_v2"]
    assert len(v2) == 3 and all(r["judge_prompt"] == "composite_v2" for r in v2)


def test_sample_paired_returns_all_when_k_exceeds_outputs():
    rows = mapping.build_rows(OUTPUTS, THREE_JUDGES)   # 2 outputs x 3 = 6
    assert len(mapping.sample_paired(rows, k_outputs=99)) == 6
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
