"""MongoDB system-of-record adapter plus an in-memory test/development repository."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
from threading import RLock
from typing import Any, Iterable, Protocol, TypeVar

from pydantic import BaseModel

from .models import RunStatus, WorkItem, WorkStatus, utcnow

T = TypeVar("T", bound=BaseModel)

COLLECTIONS = {
    "projects", "documents", "task_specs", "prompt_versions", "model_profiles",
    "tool_profiles", "ground_truth", "runs", "work_items", "extractions",
    "evaluations", "traces", "audit_events",
}


def _dump(value: BaseModel | dict[str, Any]) -> dict[str, Any]:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="python")
    return deepcopy(value)


class Repository(Protocol):
    def health(self) -> tuple[bool, str]: ...
    def put(self, collection: str, value: BaseModel | dict[str, Any]) -> dict[str, Any]: ...
    def get(self, collection: str, item_id: str) -> dict[str, Any] | None: ...
    def find(self, collection: str, query: dict[str, Any] | None = None) -> list[dict[str, Any]]: ...
    def update(self, collection: str, item_id: str, changes: dict[str, Any]) -> dict[str, Any] | None: ...
    def queue_run(self, run_id: str) -> bool: ...
    def claim_work(self, run_id: str, owner: str, lease_seconds: int) -> dict[str, Any] | None: ...
    def update_work_if_owned(self, item_id: str, owner: str, attempt: int, changes: dict[str, Any]) -> bool: ...


class InMemoryRepository:
    def __init__(self) -> None:
        self.data: dict[str, dict[str, dict[str, Any]]] = {name: {} for name in COLLECTIONS}
        self._lock = RLock()

    def health(self) -> tuple[bool, str]:
        return True, "in-memory repository available"

    def put(self, collection: str, value: BaseModel | dict[str, Any]) -> dict[str, Any]:
        doc = _dump(value)
        item_id = str(doc.get("id") or doc.get("_id"))
        if not item_id or item_id == "None":
            raise ValueError("repository documents require id")
        doc["id"] = item_id
        with self._lock:
            unique = self._unique_existing(collection, doc)
            if unique:
                item_id = unique["id"]
                doc["id"] = item_id
            self.data.setdefault(collection, {})[item_id] = deepcopy(doc)
        return deepcopy(doc)

    def _unique_existing(self, collection: str, doc: dict[str, Any]) -> dict[str, Any] | None:
        fields = {
            "work_items": ("identity",),
            "extractions": ("work_item_id",),
            "evaluations": ("extraction_id",),
            "traces": ("work_item_id", "attempt", "sequence"),
        }.get(collection)
        if not fields:
            return None
        return next((
            item for item in self.data.setdefault(collection, {}).values()
            if all(item.get(field) == doc.get(field) for field in fields)
        ), None)

    def get(self, collection: str, item_id: str) -> dict[str, Any] | None:
        doc = self.data.setdefault(collection, {}).get(str(item_id))
        return deepcopy(doc) if doc else None

    def find(self, collection: str, query: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        query = query or {}
        docs = list(self.data.setdefault(collection, {}).values())
        return [deepcopy(d) for d in docs if all(d.get(k) == v for k, v in query.items())]

    def update(self, collection: str, item_id: str, changes: dict[str, Any]) -> dict[str, Any] | None:
        with self._lock:
            current = self.data.setdefault(collection, {}).get(str(item_id))
            if current is None:
                return None
            current.update(deepcopy(changes))
            return deepcopy(current)

    def queue_run(self, run_id: str) -> bool:
        with self._lock:
            run = self.data["runs"].get(run_id)
            if not run or run.get("status") not in {
                RunStatus.PREFLIGHT_PENDING.value, RunStatus.PREFLIGHT_READY.value,
            }:
                return False
            run["status"] = RunStatus.DISPATCHING.value
            run["approved_at"] = utcnow()
            run["dispatch_started_at"] = utcnow()
            return True

    def claim_work(self, run_id: str, owner: str, lease_seconds: int) -> dict[str, Any] | None:
        now = utcnow()
        with self._lock:
            candidates = sorted(self.data["work_items"].values(), key=lambda d: d["created_at"])
            for work in candidates:
                expired = work.get("lease_expires_at") and work["lease_expires_at"] <= now
                if work["run_id"] == run_id and (
                    work["status"] == WorkStatus.PENDING.value
                    or (work["status"] == WorkStatus.LEASED.value and expired)
                ):
                    work.update({
                        "status": WorkStatus.LEASED.value,
                        "lease_owner": owner,
                        "lease_expires_at": now + timedelta(seconds=lease_seconds),
                        "started_at": work.get("started_at") or now,
                        "attempt": int(work.get("attempt", 0)) + 1,
                    })
                    return deepcopy(work)
        return None

    def update_work_if_owned(self, item_id: str, owner: str, attempt: int, changes: dict[str, Any]) -> bool:
        with self._lock:
            work = self.data["work_items"].get(item_id)
            if not work or work.get("lease_owner") != owner or int(work.get("attempt", 0)) != attempt:
                return False
            work.update(deepcopy(changes))
            return True


class MongoRepository:
    def __init__(self, uri: str, database: str) -> None:
        from pymongo import MongoClient

        self.client = MongoClient(uri, serverSelectionTimeoutMS=5000, appname="pdf-harness")
        self.db = self.client[database]
        self.ensure_indexes()

    def ensure_indexes(self) -> None:
        self.db.projects.create_index("name")
        self.db.documents.create_index([("project_id", 1), ("sha256", 1)], unique=True)
        for name in ("task_specs", "prompt_versions", "model_profiles", "tool_profiles"):
            self.db[name].create_index([("project_id", 1), ("name", 1), ("version", 1)], unique=True)
            self.db[name].create_index("content_hash")
        self.db.ground_truth.create_index(
            [("project_id", 1), ("document_id", 1), ("page_number", 1), ("revision", -1)], unique=True,
        )
        self.db.runs.create_index([("project_id", 1), ("created_at", -1)])
        self.db.work_items.create_index("identity", unique=True)
        self.db.work_items.create_index([("run_id", 1), ("status", 1), ("lease_expires_at", 1)])
        self.db.extractions.create_index("work_item_id", unique=True)
        self.db.evaluations.create_index("extraction_id", unique=True)
        trace_indexes = {index["name"]: index for index in self.db.traces.list_indexes()}
        legacy_trace = trace_indexes.get("work_item_id_1_sequence_1")
        if legacy_trace and legacy_trace.get("unique"):
            self.db.traces.drop_index("work_item_id_1_sequence_1")
        self.db.traces.create_index([("work_item_id", 1), ("attempt", 1), ("sequence", 1)], unique=True)

    def health(self) -> tuple[bool, str]:
        try:
            self.client.admin.command("ping")
            return True, f"MongoDB database {self.db.name} reachable"
        except Exception as exc:  # noqa: BLE001
            return False, f"MongoDB unavailable: {type(exc).__name__}"

    @staticmethod
    def _clean(doc: dict[str, Any] | None) -> dict[str, Any] | None:
        if doc is None:
            return None
        doc = deepcopy(doc)
        doc.pop("_id", None)
        return doc

    def put(self, collection: str, value: BaseModel | dict[str, Any]) -> dict[str, Any]:
        doc = _dump(value)
        item_id = str(doc.get("id"))
        if not item_id:
            raise ValueError("repository documents require id")
        unique_fields = {
            "work_items": ("identity",),
            "extractions": ("work_item_id",),
            "evaluations": ("extraction_id",),
            "traces": ("work_item_id", "attempt", "sequence"),
        }.get(collection)
        if unique_fields:
            filter_key = {field: doc[field] for field in unique_fields}
            existing = self.db[collection].find_one(filter_key, {"id": 1})
            if existing:
                doc["id"] = existing["id"]
            self.db[collection].replace_one(filter_key, doc, upsert=True)
        else:
            self.db[collection].replace_one({"id": item_id}, doc, upsert=True)
        return self._clean(doc) or {}

    def get(self, collection: str, item_id: str) -> dict[str, Any] | None:
        return self._clean(self.db[collection].find_one({"id": str(item_id)}))

    def find(self, collection: str, query: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        return [self._clean(d) or {} for d in self.db[collection].find(query or {}).sort("created_at", -1)]

    def update(self, collection: str, item_id: str, changes: dict[str, Any]) -> dict[str, Any] | None:
        from pymongo import ReturnDocument

        return self._clean(self.db[collection].find_one_and_update(
            {"id": str(item_id)}, {"$set": changes}, return_document=ReturnDocument.AFTER
        ))

    def queue_run(self, run_id: str) -> bool:
        result = self.db.runs.update_one(
            {"id": run_id, "status": {"$in": [
                RunStatus.PREFLIGHT_PENDING.value, RunStatus.PREFLIGHT_READY.value,
            ]}},
            {"$set": {
                "status": RunStatus.DISPATCHING.value, "approved_at": utcnow(),
                "dispatch_started_at": utcnow(),
            }},
        )
        return result.modified_count == 1

    def claim_work(self, run_id: str, owner: str, lease_seconds: int) -> dict[str, Any] | None:
        from pymongo import ReturnDocument

        now = utcnow()
        query = {
            "run_id": run_id,
            "$or": [
                {"status": WorkStatus.PENDING.value},
                {"status": WorkStatus.LEASED.value, "lease_expires_at": {"$lte": now}},
            ],
        }
        update = {
            "$set": {
                "status": WorkStatus.LEASED.value,
                "lease_owner": owner,
                "lease_expires_at": now + timedelta(seconds=lease_seconds),
                "started_at": now,
            },
            "$inc": {"attempt": 1},
        }
        doc = self.db.work_items.find_one_and_update(
            query, update, sort=[("created_at", 1)], return_document=ReturnDocument.AFTER
        )
        return self._clean(doc)

    def update_work_if_owned(self, item_id: str, owner: str, attempt: int, changes: dict[str, Any]) -> bool:
        result = self.db.work_items.update_one(
            {"id": item_id, "status": WorkStatus.LEASED.value, "lease_owner": owner, "attempt": attempt},
            {"$set": changes},
        )
        return result.modified_count == 1
