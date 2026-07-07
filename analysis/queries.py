"""
queries.py — DuckDB over the Parquet run-summaries (plan §3, the OLAP layer).

DuckDB reads the date-partitioned Parquet files directly — no warehouse, no
load step. This module exposes a `runs` view plus a few canned aggregations the
dashboard and notebooks use.

    from analysis.queries import connect, summary_by, concurrency_summary
    con = connect()
    print(summary_by(con, "backend", "reasoning_level"))
    print(concurrency_summary(con))
"""

from __future__ import annotations

from pathlib import Path

import duckdb

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_DIR = REPO_ROOT / "outputs" / "agent_runs"


def runs_glob(base_dir=None) -> str:
    base = Path(base_dir) if base_dir is not None else DEFAULT_RUNS_DIR
    return str(base / "**" / "*.parquet")


def connect(base_dir=None):
    """Return a DuckDB connection with a `runs` view over the Parquet dataset.

    Raises FileNotFoundError if no run files exist yet (so the caller gets a
    clear message instead of a confusing empty-glob SQL error).
    """
    glob = runs_glob(base_dir)
    matches = list(Path(glob.split("**")[0]).rglob("*.parquet"))
    if not matches:
        raise FileNotFoundError(
            f"no run Parquet files under {glob} — run `python -m eval ...` first"
        )
    con = duckdb.connect()
    con.execute(
        f"CREATE VIEW runs AS SELECT * FROM read_parquet('{glob}', union_by_name=true)"
    )
    return con


def summary_by(con, *dims):
    """Aggregate key metrics grouped by the given dimension columns."""
    if not dims:
        dims = ("backend",)
    cols = ", ".join(dims)
    return con.execute(
        f"""
        SELECT {cols},
               count(*)                         AS n_runs,
               avg(tokens_per_sec)              AS avg_tokens_per_sec,
               avg(wall_time_total)             AS avg_wall_s,
               avg(total_tokens)                AS avg_total_tokens,
               avg(peak_context)                AS avg_peak_context,
               avg(CAST(save_success AS DOUBLE)) AS save_success_rate,
               avg(selection_accuracy)          AS avg_selection_accuracy,
               sum(n_tool_errors)               AS tool_errors
        FROM runs
        GROUP BY {cols}
        ORDER BY {cols}
        """
    ).df()


def concurrency_summary(con):
    """Sequential-vs-parallel view: throughput & GPU pressure by concurrency."""
    return con.execute(
        """
        SELECT concurrency,
               count(*)                       AS n_runs,
               avg(tokens_per_sec)            AS avg_tokens_per_sec,
               avg(wall_time_total)           AS avg_wall_s,
               avg(requests_running_end)      AS avg_requests_running,
               avg(gpu_cache_usage_end)       AS avg_gpu_cache_usage,
               sum(n_tool_errors)             AS tool_errors
        FROM runs
        GROUP BY concurrency
        ORDER BY concurrency
        """
    ).df()


def infra_summary(con):
    """Infra-comparison view: throughput & GPU pressure by framework × GPU × concurrency.

    The richer version of concurrency_summary — shows whether a serving engine
    actually batches under load (requests_running rising with concurrency) and how
    that differs across GPU types. Uses columns added in Part 3; safe on older
    Parquet thanks to union_by_name (missing cols read as NULL).
    """
    return con.execute(
        """
        SELECT framework,
               gpu_type,
               concurrency,
               count(*)                       AS n_runs,
               avg(tokens_per_sec)            AS avg_tokens_per_sec,
               avg(wall_time_total)           AS avg_wall_s,
               avg(requests_running_end)      AS avg_requests_running,
               avg(gpu_cache_usage_end)       AS avg_gpu_cache_usage,
               sum(n_tool_errors)             AS tool_errors
        FROM runs
        GROUP BY framework, gpu_type, concurrency
        ORDER BY framework, gpu_type, concurrency
        """
    ).df()
