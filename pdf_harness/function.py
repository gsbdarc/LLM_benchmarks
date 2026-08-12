"""Private HTTP function that atomically queues a run and starts a Cloud Run Job."""

from __future__ import annotations

from uuid import UUID

import functions_framework

from .bootstrap import build_context
from .models import RunStatus


def _canonical_id(value) -> str | None:
    try:
        parsed = UUID(str(value))
    except (ValueError, TypeError, AttributeError):
        return None
    canonical = str(parsed)
    return canonical if str(value).lower() == canonical else None


@functions_framework.http
def dispatch_run(request):
    if request.method != "POST":
        return {"error": "POST required"}, 405
    body = request.get_json(silent=True) or {}
    run_id = _canonical_id(body.get("run_id"))
    if not run_id:
        return {"error": "valid run_id required"}, 400

    context = build_context()
    run = context.repo.get("runs", run_id)
    if not run:
        return {"error": "run not found"}, 404
    status = run.get("status")
    if status in {RunStatus.QUEUED.value, RunStatus.RUNNING.value} and run.get("cloud_execution_name"):
        return {"run_id": run_id, "status": status, "execution": run.get("cloud_execution_name")}, 202
    if status == RunStatus.DISPATCH_UNKNOWN.value:
        context.repo.update("runs", run_id, {"status": RunStatus.DISPATCHING.value})
    elif not context.repo.queue_run(run_id):
        return {"error": f"run cannot be queued from status {status}"}, 409
    try:
        execution = context.dispatcher.dispatch(run_id)
        context.repo.update("runs", run_id, {"cloud_execution_name": execution, "status": RunStatus.QUEUED.value})
    except Exception as exc:  # noqa: BLE001
        context.repo.update("runs", run_id, {
            "status": RunStatus.DISPATCH_UNKNOWN.value,
            "error": f"dispatch outcome unknown: {type(exc).__name__}: {str(exc)[:300]}",
        })
        return {"error": "Cloud Run Job dispatch failed"}, 502
    return {"run_id": run_id, "status": RunStatus.QUEUED.value, "execution": execution}, 202
