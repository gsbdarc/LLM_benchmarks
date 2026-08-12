"""Standalone Gen-2 Cloud Function dispatcher.

IAM must require authentication. This function contains only dispatch logic so its
deployment package does not need the full Streamlit/agent dependency tree.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from uuid import UUID

import functions_framework
from google.cloud import run_v2
from pymongo import MongoClient, ReturnDocument

QUEUEABLE = ["preflight_pending", "preflight_ready"]


def _db():
    uri = os.environ["HARNESS_MONGO_URI"]
    return MongoClient(uri, serverSelectionTimeoutMS=5000, appname="pdf-harness-dispatcher")[
        os.getenv("HARNESS_MONGO_DB", "pdf_extraction_harness")
    ]


def _job_name() -> str:
    return (
        f"projects/{os.environ['HARNESS_GCP_PROJECT']}/locations/"
        f"{os.getenv('HARNESS_GCP_REGION', 'us-west1')}/jobs/{os.environ['HARNESS_CLOUD_RUN_JOB']}"
    )


def _canonical_id(value) -> str | None:
    try:
        parsed = UUID(str(value))
    except (ValueError, TypeError, AttributeError):
        return None
    canonical = str(parsed)
    return canonical if str(value).lower() == canonical else None


def _start_job(env_name: str, value: str) -> str:
    overrides = run_v2.RunJobRequest.Overrides(
        container_overrides=[run_v2.RunJobRequest.Overrides.ContainerOverride(
            env=[run_v2.EnvVar(name=env_name, value=value)]
        )]
    )
    operation = run_v2.JobsClient().run_job(
        request=run_v2.RunJobRequest(name=_job_name(), overrides=overrides)
    )
    return operation.operation.name


def _dispatch_document(db, document_id: str):
    current = db.documents.find_one({"id": document_id}, {"render_status": 1})
    if not current:
        return {"error": "document not found"}, 404
    if current.get("render_status") in {"queued", "running", "succeeded"}:
        return {"document_id": document_id, "status": current["render_status"]}, 202
    claimed = db.documents.update_one(
        {"id": document_id, "render_status": {"$in": ["pending", "failed", "dispatch_unknown"]}},
        {"$set": {"render_status": "dispatching", "error": None}},
    )
    if claimed.modified_count != 1:
        return {"error": f"document cannot be dispatched from status {current.get('render_status')}"}, 409
    try:
        execution = _start_job("HARNESS_DOCUMENT_ID", document_id)
        db.documents.update_one(
            {"id": document_id},
            {"$set": {"render_status": "queued", "render_execution_name": execution}},
        )
    except Exception as exc:  # noqa: BLE001
        db.documents.update_one({"id": document_id}, {"$set": {
            "render_status": "dispatch_unknown",
            "error": f"render dispatch outcome unknown: {type(exc).__name__}: {str(exc)[:300]}",
        }})
        return {"error": "render dispatch failed"}, 502
    return {"document_id": document_id, "status": "queued", "execution": execution}, 202


@functions_framework.http
def dispatch_run(request):
    if request.method != "POST":
        return {"error": "POST required"}, 405
    body = request.get_json(silent=True) or {}
    db = _db()
    document_id = _canonical_id(body.get("document_id"))
    if document_id:
        return _dispatch_document(db, document_id)
    run_id = _canonical_id(body.get("run_id"))
    if not run_id:
        return {"error": "valid run_id or document_id required"}, 400
    current = db.runs.find_one({"id": run_id}, {"status": 1, "cloud_execution_name": 1})
    if not current:
        return {"error": "run not found"}, 404
    if current.get("status") in {"queued", "running"} and current.get("cloud_execution_name"):
        return {"run_id": run_id, "status": current["status"], "execution": current.get("cloud_execution_name")}, 202
    now = datetime.now(timezone.utc)
    stale = now - timedelta(minutes=2)
    queued = db.runs.find_one_and_update(
        {"id": run_id, "$or": [
            {"status": {"$in": QUEUEABLE}},
            {"status": "dispatch_unknown"},
            {"status": "dispatching", "dispatch_started_at": {"$lte": stale}},
            {"status": "queued", "cloud_execution_name": None},
        ]},
        {"$set": {"status": "dispatching", "approved_at": now, "dispatch_started_at": now}},
        return_document=ReturnDocument.AFTER,
    )
    if not queued:
        return {"error": f"run cannot be queued from status {current.get('status')}"}, 409
    try:
        execution = _start_job("HARNESS_RUN_ID", run_id)
        db.runs.update_one({"id": run_id}, {"$set": {"cloud_execution_name": execution, "status": "queued"}})
    except Exception as exc:  # noqa: BLE001
        db.runs.update_one({"id": run_id}, {"$set": {
            "status": "dispatch_unknown", "error": f"dispatch outcome unknown: {type(exc).__name__}: {str(exc)[:300]}",
        }})
        return {"error": "Cloud Run Job dispatch failed"}, 502
    return {"run_id": run_id, "status": "queued", "execution": execution}, 202
