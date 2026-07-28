"""
export_runs.py — materialize the central agentic_runs collection to a Parquet cache.

The "living" dashboard's cache step (tokenviz pattern): the self-rescheduling Slurm
refresh (agent_eval/scripts/refresh_dashboard.slurm) dumps the team-central
`agentic_runs` Mongo collection here, then build_dashboard renders the standalone HTML
from it. This keeps build_dashboard's Parquet/DuckDB path unchanged — `agentic_runs`
mirrors the exact flat schema the local sink writes (tools.save_run_row stores the same
row), so it drops straight in.

    python -m analysis.export_runs --out outputs/dashboard_cache
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Union

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "outputs" / "dashboard_cache"


def export(out_dir: Union[str, Path] = DEFAULT_OUT) -> Path:
    """Dump every agentic_runs doc to a Parquet file under out_dir/data/ and return it.

    Drops Mongo-only fields (`_id`, `stored_at`). Written under a `data/` subdir so
    build_dashboard's `base_dir/**/*.parquet` glob (queries.runs_glob) picks it up.
    """
    from agent_eval import tools  # lazy: keeps Mongo out of import time

    db = tools.get_db()
    docs = list(db[tools.AGENTIC_RUNS_COLL].find({}, {"_id": 0}))
    df = pd.DataFrame(docs)
    df = df.drop(columns=[c for c in ("stored_at",) if c in df.columns], errors="ignore")

    part = Path(out_dir) / "data"
    part.mkdir(parents=True, exist_ok=True)
    path = part / "agentic_runs.parquet"
    df.to_parquet(path, engine="pyarrow", index=False)
    return path


def main(argv: Optional[list[str]] = None) -> None:
    """CLI: export agentic_runs to a Parquet cache dir for the dashboard build."""
    p = argparse.ArgumentParser(prog="analysis.export_runs")
    p.add_argument("--out", default=str(DEFAULT_OUT), help="dir to write the Parquet cache")
    args = p.parse_args(argv)
    path = export(args.out)
    print(f"Exported {len(pd.read_parquet(path))} agentic_runs rows -> {path}")


if __name__ == "__main__":
    main()
