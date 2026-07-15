"""
mapping.py — the eval-run registry (agentic analogue of scripts/3_create_mapping.py).

An append-only CSV (`inputs/eval_mapping.csv`) keyed by a stable integer `eval_id`.
One row = one evaluation job = one existing llm_output judged by one agent/judge
config. Because the first run uses a single judge config, the registry is 1:1 with
the sampled outputs; the judge columns make future prompt/model sweeps a pure data
change (add judge configs -> the cross-product grows, existing eval_ids stay put).

Everything here is pure (no DB, no module globals) so it is unit-tested directly.
`scripts/3b_create_eval_mapping.py` is the thin CLI that pulls outputs from Mongo
and calls these; the array worker (`python -m agent_eval --eval-mapping … --row N`) reads
one row via `read_mapping_row`.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any

# Full row schema. eval_id is the surrogate key; the rest identify the output and
# the judge config that evaluates it.
FIELDS = [
    "eval_id", "task_id", "run_id", "benchmark_id", "model_id",
    "judge_backend", "judge_model", "judge_prompt",
]
# Columns that define uniqueness (everything but the surrogate eval_id).
KEY_FIELDS = [
    "task_id", "run_id", "benchmark_id", "model_id",
    "judge_backend", "judge_model", "judge_prompt",
]


def build_rows(
    outputs: list[dict[str, Any]], judge_configs: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Cross each output with each judge config → list of row dicts (no eval_id yet).

    `outputs`: dicts with task_id/run_id/benchmark_id/model_id (as from
    metric_tools.list_pending_outputs). `judge_configs`: dicts with
    judge_backend/judge_model/judge_prompt.
    """
    rows = []
    for o in outputs:
        for j in judge_configs:
            rows.append({
                "task_id": str(o["task_id"]),
                "run_id": o.get("run_id"),
                "benchmark_id": str(o["benchmark_id"]),
                "model_id": None if o.get("model_id") is None else str(o.get("model_id")),
                "judge_backend": j["judge_backend"],
                "judge_model": j["judge_model"],
                "judge_prompt": j["judge_prompt"],
            })
    return rows


def _key(row: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(row.get(k)) for k in KEY_FIELDS)


def dedupe_and_assign(
    existing: list[dict[str, Any]], candidates: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Return the candidate rows not already in `existing`, each given a fresh
    sequential eval_id continuing past the max existing id. Idempotent."""
    seen = {_key(r) for r in existing}
    next_id = max((int(r["eval_id"]) for r in existing), default=-1) + 1
    out = []
    for c in candidates:
        k = _key(c)
        if k in seen:
            continue
        seen.add(k)
        out.append({"eval_id": next_id, **c})
        next_id += 1
    return out


def stratified_sample(
    rows: list[dict[str, Any]],
    n: int | None,
    key: str = "benchmark_id",
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Deterministically sample ~n rows spread evenly across distinct `key` values."""
    if n is None or n >= len(rows):
        return list(rows)
    rng = random.Random(seed)
    groups: dict = {}
    for r in rows:
        groups.setdefault(str(r.get(key)), []).append(r)
    per = max(1, n // len(groups))
    picked = []
    for g in sorted(groups):
        items = list(groups[g])
        rng.shuffle(items)
        picked.extend(items[:per])
    rng.shuffle(picked)
    return picked[:n]


# ── CSV I/O ──────────────────────────────────────────────────────────

def read_csv(path: str | Path) -> list[dict[str, Any]]:
    """Read a mapping CSV into a list of row dicts (empty list if absent/empty)."""
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return []
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(
    path: str | Path, rows: list[dict[str, Any]], fields: list[str] = FIELDS
) -> None:
    """Overwrite `path` with `rows`, writing a header and only the `fields` columns."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})


def append_csv(
    path: str | Path, rows: list[dict[str, Any]], fields: list[str] = FIELDS
) -> None:
    """Append `rows` to `path`, writing the header first only if the file is new/empty."""
    p = Path(path)
    exists = p.exists() and p.stat().st_size > 0
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})


def read_mapping_row(path: str | Path, index: int) -> dict[str, Any]:
    """Return the row dict at 0-based `index` (header excluded)."""
    rows = read_csv(path)
    if index < 0 or index >= len(rows):
        raise IndexError(f"row {index} out of range (mapping {path} has {len(rows)} rows)")
    return rows[index]


def coerce_run_id(value: Any) -> int | None:
    """CSV reads everything as strings; map an empty/None run_id back to None, else int."""
    if value in (None, "", "None"):
        return None
    return int(value)
