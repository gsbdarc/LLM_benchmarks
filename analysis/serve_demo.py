"""Serve the date-fix demo page live on Yen.

Sibling of `serve_dashboard.py`, for the OTHER task — the pairing mirrors the builders:

    build_dashboard.py      serve_dashboard.py   # metric-eval routing   :8787
    build_date_fix_demo.py  serve_demo.py        # date repair           :8788

This is the command to hand a reviewer. `build_date_fix_demo --open` writes a 985 KB file
and calls `webbrowser.open()`, which does nothing on a headless login node; serving it over
a forwarded port is the only way to actually look at the page from a laptop.

    python -m analysis.serve_demo                 # http://127.0.0.1:8788
    python -m analysis.serve_demo --port 9000

Renders from Mongo on each page load (behind a short TTL cache), so you see *current* runs.
Needs `MONGO_DB_USERNAME` / `MONGO_DB_PASSWORD` in `.env`, and at least one `date_fix_v1`
batch in `agentic_runs` — with no runs the page is a plain-text 503 telling you to run the
batch, not a traceback. Strictly read-only. Uses Starlette + uvicorn, same as the dashboard
server.
"""
from __future__ import annotations

import argparse
import getpass
import socket
import time

import uvicorn
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.responses import HTMLResponse, PlainTextResponse
from starlette.routing import Route

from analysis import build_date_fix_demo

TTL_SECONDS = 30.0
_cache: dict[str, object] = {"at": 0.0, "html": None}


def _render_live() -> str:
    """Join Mongo + the work lists into the demo HTML.

    `build_snapshot` exits with a message when there are no date-fix runs. That is right
    for a CLI and fatal for a server, and `SystemExit` is a BaseException whose passage
    through the threadpool we would rather not depend on — so re-raise it as an ordinary
    error the route can turn into a 503.
    """
    try:
        snapshot = build_date_fix_demo.build_snapshot()
    except SystemExit as e:
        raise RuntimeError(str(e)) from e
    return build_date_fix_demo.render_html(snapshot)


async def demo(request):
    now = time.monotonic()
    if _cache["html"] is None or now - float(_cache["at"]) > TTL_SECONDS:
        try:
            # Mongo/pandas are blocking — keep them off the event loop.
            _cache["html"] = await run_in_threadpool(_render_live)
        except RuntimeError as e:
            return PlainTextResponse(f"cannot build the demo: {e}\n", status_code=503)
        _cache["at"] = now
    return HTMLResponse(_cache["html"])


async def health(request):
    return PlainTextResponse("ok")


app = Starlette(routes=[Route("/", demo), Route("/healthz", health)])


def forward_hint(port: int) -> str:
    """The ssh -L line for this host, so a reviewer can copy-paste it.

    `socket.getfqdn()` returns the internal `yen4.yen.sunet`, which is not reachable from
    a laptop; the short name plus `.stanford.edu` is (yen1-yen8.stanford.edu resolve).
    """
    host = socket.gethostname().split(".")[0]
    return (f"    ssh -N -L {port}:localhost:{port} "
            f"{getpass.getuser()}@{host}.stanford.edu")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="analysis.serve_demo")
    p.add_argument("--host", default="127.0.0.1", help="bind address (localhost only)")
    p.add_argument("--port", type=int, default=8788)
    args = p.parse_args(argv)
    # flush: uvicorn logs to stderr, so an unflushed stdout banner lands after it in a log file.
    print(f"\ndate-fix demo:  http://localhost:{args.port}\n\n"
          f"  From your laptop, forward the port first:\n"
          f"{forward_hint(args.port)}\n\n"
          f"  (VS Code's Ports panel does this for you — then just open the URL.)\n",
          flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
