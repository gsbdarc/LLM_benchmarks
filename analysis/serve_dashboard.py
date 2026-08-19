"""Serve the agentic-eval dashboard live on Yen.

Renders the dashboard HTML from the central `agentic_runs` Mongo collection on each
page load (behind a short TTL cache), so you view *current* data in a browser instead
of downloading a static file. Reuses the same pieces as the one-off build:
`export_runs.export` (Mongo -> Parquet) -> `build_dashboard.build_snapshot` (DuckDB) ->
`build_dashboard.render_html`.

    python -m analysis.serve_dashboard            # http://127.0.0.1:8787
    python -m analysis.serve_dashboard --port 9000

Bound to localhost; view it through VS Code's forwarded port / Simple Browser. Strictly
read-only — it only reads agentic_runs and never writes Mongo. Uses Starlette + uvicorn,
which ship with `mcp` (no new dependency).
"""
from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

import uvicorn
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.responses import HTMLResponse, PlainTextResponse
from starlette.routing import Route

from analysis import build_dashboard, export_runs

TTL_SECONDS = 30.0
# One temp cache dir for the process lifetime; export overwrites the Parquet in place.
_CACHE_DIR = Path(tempfile.mkdtemp(prefix="dashboard_cache_"))
_cache: dict[str, object] = {"at": 0.0, "html": None}


def _render_live() -> str:
    """Re-export agentic_runs -> Parquet, then render the dashboard HTML (no summaries)."""
    export_runs.export(_CACHE_DIR)
    snapshot = build_dashboard.build_snapshot(_CACHE_DIR, summarize=False)
    return build_dashboard.render_html(snapshot)


async def dashboard(request):
    now = time.monotonic()
    if _cache["html"] is None or now - float(_cache["at"]) > TTL_SECONDS:
        # Mongo/DuckDB/pandas are blocking — keep them off the event loop.
        _cache["html"] = await run_in_threadpool(_render_live)
        _cache["at"] = now
    return HTMLResponse(_cache["html"])


async def health(request):
    return PlainTextResponse("ok")


app = Starlette(routes=[Route("/", dashboard), Route("/healthz", health)])


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="analysis.serve_dashboard")
    p.add_argument("--host", default="127.0.0.1", help="bind address (localhost only)")
    p.add_argument("--port", type=int, default=8787)
    args = p.parse_args(argv)
    print(f"Serving live dashboard on http://{args.host}:{args.port}  (cache: {_CACHE_DIR})")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
