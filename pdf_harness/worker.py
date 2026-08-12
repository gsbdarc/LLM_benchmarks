"""Lease-based extraction worker used locally and by Cloud Run Jobs."""

from __future__ import annotations

import os
import socket
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any

from .llm import ExtractionClient, estimated_cost
from .models import (
    Document, Evaluation, Extraction, GroundTruth, ModelProfileVersion, PreflightResult,
    PromptVersion, Run, RunStatus, TaskSpecVersion, TraceEvent, WorkItem, WorkStatus,
    utcnow,
)
from .observability import NullObserver, Observer
from .repository import Repository
from .scoring import score_record
from .secrets import SecretResolver
from .storage import ArtifactStore
from .tools import invoke


def _safe_payload(value: Any) -> Any:
    """Bound traces and redact obvious credential-shaped keys."""
    if isinstance(value, dict):
        return {
            key: "[REDACTED]" if any(token in key.lower() for token in ("key", "secret", "password", "token")) else _safe_payload(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_safe_payload(item) for item in value[:100]]
    if isinstance(value, str) and len(value) > 4000:
        return value[:4000] + "...[truncated]"
    return value


class HarnessWorker:
    def __init__(
        self, repo: Repository, artifacts: ArtifactStore, secrets: SecretResolver,
        llm: ExtractionClient, concurrency: int = 4, max_attempts: int = 3,
        lease_seconds: int = 900, observer: Observer | None = None,
    ) -> None:
        self.repo = repo
        self.artifacts = artifacts
        self.secrets = secrets
        self.llm = llm
        self.concurrency = concurrency
        self.max_attempts = max_attempts
        self.lease_seconds = lease_seconds
        self.observer = observer or NullObserver()
        self.owner = f"{socket.gethostname()}:{os.getpid()}"

    def process_run(self, run_id: str) -> Run:
        raw = self.repo.get("runs", run_id)
        if not raw:
            raise KeyError(f"run {run_id} not found")
        run = Run.model_validate(raw)
        if run.status in {RunStatus.CANCELLED, RunStatus.COMPLETED, RunStatus.COMPLETED_WITH_ERRORS}:
            return run
        self.repo.update("runs", run_id, {"status": RunStatus.RUNNING.value, "started_at": run.started_at or utcnow()})

        while True:
            claimed: list[dict[str, Any]] = []
            for _ in range(self.concurrency):
                work = self.repo.claim_work(run_id, self.owner, self.lease_seconds)
                if not work:
                    break
                claimed.append(work)
            if not claimed:
                current = self.repo.get("runs", run_id) or {}
                if current.get("status") == RunStatus.CANCEL_REQUESTED.value:
                    self._cancel_pending(run_id)
                    break
                if self._ensure_aggregation_work(run_id):
                    self._refresh_progress(run_id)
                    continue
                remaining = self.repo.find("work_items", {"run_id": run_id})
                if any(item["status"] in {WorkStatus.PENDING.value, WorkStatus.LEASED.value} for item in remaining):
                    time.sleep(1)
                    continue
                break
            with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
                futures = [pool.submit(self._process_work, WorkItem.model_validate(work)) for work in claimed]
                for future in as_completed(futures):
                    future.result()
            current = self.repo.get("runs", run_id) or {}
            if current.get("status") == RunStatus.CANCEL_REQUESTED.value:
                self._cancel_pending(run_id)
                break
            self._refresh_progress(run_id)

        return self._finish(run_id)

    def _trace(self, work: WorkItem, sequence: int, event_type: str, name: str, payload: dict[str, Any], duration: float | None = None) -> None:
        event = TraceEvent(
            run_id=work.run_id, work_item_id=work.id, sequence=sequence,
            attempt=work.attempt,
            event_type=event_type, name=name, payload=_safe_payload(payload), duration_seconds=duration,
        )
        self.repo.put("traces", event)

    def _process_work(self, work: WorkItem) -> None:
        try:
            offset = 0
            if work.attempt > 1:
                self._trace(work, 0, "retry", "work_item_retry", {"attempt": work.attempt})
                offset = 1
            existing = self.repo.find("extractions", {"work_item_id": work.id})
            if existing:
                run = Run.model_validate(self.repo.get("runs", work.run_id))
                spec = TaskSpecVersion.model_validate(self.repo.get("task_specs", run.snapshot.task_spec_version_id))
                self._score_if_labeled(run, spec, Extraction.model_validate(existing[0]))
                self.repo.update_work_if_owned(work.id, self.owner, work.attempt, {
                    "status": WorkStatus.SUCCEEDED.value, "completed_at": utcnow(),
                    "lease_owner": None, "lease_expires_at": None, "error": None,
                })
                return
            if work.stage.value != "extract":
                if work.stage.value == "aggregate":
                    self._process_aggregate(work)
                    return
                raise ValueError(f"unsupported work stage {work.stage.value}")
            run = Run.model_validate(self.repo.get("runs", work.run_id))
            document = Document.model_validate(self.repo.get("documents", work.document_id))
            spec = TaskSpecVersion.model_validate(self.repo.get("task_specs", run.snapshot.task_spec_version_id))
            profile = ModelProfileVersion.model_validate(self.repo.get("model_profiles", work.model_profile_version_id or ""))
            prompt = PromptVersion.model_validate(self.repo.get("prompt_versions", work.prompt_version_id or ""))
            page_uri = document.page_uris.get(str(work.page_number or 0))
            if not page_uri:
                raise ValueError(f"document has no rendered page {work.page_number}")
            image = self.artifacts.get(page_uri)
            self._trace(work, offset, "model_request", profile.model_id, {
                "document_id": document.id, "page_number": work.page_number,
                "prompt_hash": prompt.content_hash, "schema_hash": spec.content_hash,
            })
            result = self.llm.extract(
                profile, prompt, self.secrets.get(profile.secret_ref), image, "image/png",
                spec.json_schema, {"document_name": document.name, "page_number": work.page_number},
            )
            self._trace(work, offset + 1, "model_response", profile.model_id, {
                "output": result.output, "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
            }, result.latency_seconds)
            tool_raw = self.repo.get("tool_profiles", run.snapshot.tool_profile_version_id or "")
            if not tool_raw:
                raise ValueError("snapshotted tool profile is missing")
            allowed_tools = tool_raw.get("allowed_tools", [])
            validation = invoke(
                "validate_schema", allowed_tools, value=result.output or {}, schema=spec.json_schema,
            )
            self._trace(work, offset + 2, "tool_call", "validate_schema", {
                "schema_hash": spec.content_hash, "tool_profile_version_id": tool_raw["id"],
                "allowed_tools": allowed_tools,
            })
            self._trace(work, offset + 3, "tool_result", "validate_schema", validation)
            repair_attempts = 0
            original_raw_response = result.raw_response
            repair_response = None
            if not validation["valid"] and hasattr(self.llm, "repair"):
                repair_attempts = 1
                self._trace(work, offset + 4, "retry", "structured_output_repair", {
                    "validation_errors": validation["errors"][:20],
                })
                repaired = self.llm.repair(
                    profile, self.secrets.get(profile.secret_ref), result.raw_response,
                    spec.json_schema, {"document_name": document.name, "page_number": work.page_number},
                )
                result.output = repaired.output
                repair_response = repaired.raw_response
                result.input_tokens += repaired.input_tokens
                result.output_tokens += repaired.output_tokens
                result.latency_seconds += repaired.latency_seconds
                validation = invoke(
                    "validate_schema", allowed_tools, value=result.output or {}, schema=spec.json_schema,
                )
                self._trace(work, offset + 5, "tool_result", "validate_schema_after_repair", validation)
            extraction = Extraction(
                work_item_id=work.id, run_id=work.run_id, project_id=work.project_id,
                document_id=work.document_id, page_number=work.page_number,
                model_profile_version_id=profile.id, prompt_version_id=prompt.id,
                source_prompt_version_id=work.source_prompt_version_id or prompt.id,
                output=result.output, raw_response=original_raw_response,
                repair_response=repair_response,
                schema_valid=validation["valid"], validation_errors=validation["errors"],
                repair_attempts=repair_attempts,
                input_tokens=result.input_tokens, output_tokens=result.output_tokens,
                estimated_cost_usd=estimated_cost(profile, result.input_tokens, result.output_tokens),
                latency_seconds=result.latency_seconds,
            )
            self.repo.put("extractions", extraction)
            self._score_if_labeled(run, spec, extraction)
            self.observer.record("extraction", {
                "project_id": work.project_id, "run_id": work.run_id,
                "work_item_id": work.id, "document_id": work.document_id,
                "page_number": work.page_number, "model_profile_version_id": profile.id,
                "prompt_version_id": prompt.id, "prompt_hash": prompt.content_hash,
                "schema_hash": spec.content_hash, "schema_valid": extraction.schema_valid,
                "latency_seconds": extraction.latency_seconds,
                "input_tokens": extraction.input_tokens, "output_tokens": extraction.output_tokens,
                "estimated_cost_usd": extraction.estimated_cost_usd,
                "tool_sequence": ["validate_schema"],
            })
            if not self.repo.update_work_if_owned(work.id, self.owner, work.attempt, {
                "status": WorkStatus.SUCCEEDED.value, "completed_at": utcnow(),
                "lease_owner": None, "lease_expires_at": None, "error": None,
            }):
                raise RuntimeError("work lease was lost before completion")
        except Exception as exc:  # noqa: BLE001
            message = f"{type(exc).__name__}: {str(exc)[:500]}"
            self.observer.record("work_item_error", {
                "project_id": work.project_id, "run_id": work.run_id,
                "work_item_id": work.id, "attempt": work.attempt,
                "error_type": type(exc).__name__,
            })
            self._trace(work, 99, "error", "work_item_failed", {"error": message})
            terminal = work.attempt >= self.max_attempts
            self.repo.update_work_if_owned(work.id, self.owner, work.attempt, {
                "status": WorkStatus.FAILED.value if terminal else WorkStatus.PENDING.value,
                "error": message, "lease_owner": None, "lease_expires_at": None,
                "completed_at": utcnow() if terminal else None,
            })

    def _ensure_aggregation_work(self, run_id: str) -> bool:
        run = Run.model_validate(self.repo.get("runs", run_id))
        if run.is_preflight:
            return False
        spec = TaskSpecVersion.model_validate(self.repo.get("task_specs", run.snapshot.task_spec_version_id))
        if not spec.aggregation_enabled or not spec.aggregation_prompt_version_id:
            return False
        current = self.repo.find("work_items", {"run_id": run_id})
        extraction_items = [item for item in current if item["stage"] == "extract"]
        if any(item["status"] not in {WorkStatus.SUCCEEDED.value, WorkStatus.FAILED.value, WorkStatus.CANCELLED.value} for item in extraction_items):
            return False
        if any(item["stage"] == "aggregate" for item in current):
            return False
        created = []
        for document_id in run.snapshot.document_ids:
            for model_id in run.snapshot.model_profile_version_ids:
                for source_prompt_id in run.snapshot.prompt_version_ids:
                    successful = self.repo.find("extractions", {
                        "run_id": run_id, "document_id": document_id,
                        "model_profile_version_id": model_id,
                        "source_prompt_version_id": source_prompt_id,
                    })
                    if not successful:
                        continue
                    work = WorkItem(
                        run_id=run_id, project_id=run.project_id, stage="aggregate",
                        document_id=document_id, model_profile_version_id=model_id,
                        prompt_version_id=spec.aggregation_prompt_version_id,
                        source_prompt_version_id=source_prompt_id,
                    )
                    self.repo.put("work_items", work)
                    created.append(work)
        if created:
            self.repo.update("runs", run_id, {"total_items": run.total_items + len(created)})
        return bool(created)

    def _process_aggregate(self, work: WorkItem) -> None:
        run = Run.model_validate(self.repo.get("runs", work.run_id))
        document = Document.model_validate(self.repo.get("documents", work.document_id))
        spec = TaskSpecVersion.model_validate(self.repo.get("task_specs", run.snapshot.task_spec_version_id))
        profile = ModelProfileVersion.model_validate(self.repo.get("model_profiles", work.model_profile_version_id or ""))
        prompt = PromptVersion.model_validate(self.repo.get("prompt_versions", work.prompt_version_id or ""))
        page_extractions = sorted(self.repo.find("extractions", {
            "run_id": run.id, "document_id": document.id,
            "model_profile_version_id": profile.id,
            "source_prompt_version_id": work.source_prompt_version_id,
        }), key=lambda item: int(item.get("page_number") or 0))
        page_outputs = [
            {"page_number": item["page_number"], "output": item.get("output"), "schema_valid": item.get("schema_valid")}
            for item in page_extractions if item.get("page_number") is not None
        ]
        if not page_outputs:
            raise ValueError("document aggregation has no page outputs")
        self._trace(work, 0, "model_request", "document_aggregation", {
            "document_id": document.id, "source_pages": [item["page_number"] for item in page_outputs],
            "source_prompt_version_id": work.source_prompt_version_id,
        })
        result = self.llm.aggregate(
            profile, prompt, self.secrets.get(profile.secret_ref), page_outputs,
            spec.json_schema, {"document_name": document.name},
        )
        tool_raw = self.repo.get("tool_profiles", run.snapshot.tool_profile_version_id or "")
        if not tool_raw:
            raise ValueError("snapshotted tool profile is missing")
        validation = invoke(
            "validate_schema", tool_raw.get("allowed_tools", []),
            value=result.output or {}, schema=spec.json_schema,
        )
        self._trace(work, 1, "model_response", "document_aggregation", {
            "output": result.output, "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
        }, result.latency_seconds)
        self._trace(work, 2, "tool_result", "validate_schema", validation)
        extraction = Extraction(
            work_item_id=work.id, run_id=run.id, project_id=run.project_id,
            document_id=document.id, page_number=None,
            model_profile_version_id=profile.id, prompt_version_id=prompt.id,
            source_prompt_version_id=work.source_prompt_version_id,
            output=result.output, raw_response=result.raw_response,
            schema_valid=validation["valid"], validation_errors=validation["errors"],
            input_tokens=result.input_tokens, output_tokens=result.output_tokens,
            estimated_cost_usd=estimated_cost(profile, result.input_tokens, result.output_tokens),
            latency_seconds=result.latency_seconds,
        )
        self.repo.put("extractions", extraction)
        self._score_if_labeled(run, spec, extraction)
        if not self.repo.update_work_if_owned(work.id, self.owner, work.attempt, {
            "status": WorkStatus.SUCCEEDED.value, "completed_at": utcnow(),
            "lease_owner": None, "lease_expires_at": None, "error": None,
        }):
            raise RuntimeError("aggregation work lease was lost before completion")

    def _score_if_labeled(self, run: Run, spec: TaskSpecVersion, extraction: Extraction) -> None:
        if run.snapshot.mode.value != "benchmark" or extraction.output is None:
            return
        label_id = run.snapshot.ground_truth_ids.get(
            f"{extraction.document_id}:{extraction.page_number}"
        ) or run.snapshot.ground_truth_ids.get(f"{extraction.document_id}:document")
        if not label_id:
            return
        label_raw = self.repo.get("ground_truth", label_id)
        if not label_raw:
            raise ValueError("snapshotted ground-truth revision is missing")
        label = GroundTruth.model_validate(label_raw)
        if not label.approved or label.validation_errors:
            return
        deterministic = score_record(extraction.output, label.values, spec.json_schema, spec.field_rules)
        llm_judge = None
        if run.snapshot.llm_judge_enabled:
            try:
                judge_profile = ModelProfileVersion.model_validate(self.repo.get(
                    "model_profiles", run.snapshot.judge_model_profile_version_id or ""
                ))
                judge_prompt = PromptVersion.model_validate(self.repo.get(
                    "prompt_versions", run.snapshot.judge_prompt_version_id or ""
                ))
                judged = self.llm.judge(
                    judge_profile, judge_prompt, self.secrets.get(judge_profile.secret_ref),
                    extraction.output, label.values, deterministic,
                    {
                        "document_name": extraction.document_id,
                        "page_number": extraction.page_number,
                    },
                )
                llm_judge = {
                    "status": "completed" if judged.output else "invalid_response",
                    "result": judged.output,
                    "raw_response": judged.raw_response,
                    "model_profile_version_id": judge_profile.id,
                    "prompt_version_id": judge_prompt.id,
                    "input_tokens": judged.input_tokens,
                    "output_tokens": judged.output_tokens,
                    "latency_seconds": judged.latency_seconds,
                    "estimated_cost_usd": estimated_cost(
                        judge_profile, judged.input_tokens, judged.output_tokens,
                    ),
                }
            except Exception as exc:  # noqa: BLE001 — judge is experimental, deterministic scoring remains authoritative
                llm_judge = {
                    "status": "error", "error_type": type(exc).__name__,
                    "error": str(exc)[:300],
                }
        self.repo.put("evaluations", Evaluation(
            run_id=run.id, extraction_id=extraction.id,
            deterministic=deterministic, llm_judge=llm_judge,
        ))

    def _refresh_progress(self, run_id: str) -> None:
        items = self.repo.find("work_items", {"run_id": run_id})
        extractions = self.repo.find("extractions", {"run_id": run_id})
        evaluations = self.repo.find("evaluations", {"run_id": run_id})
        judges = [item.get("llm_judge") or {} for item in evaluations]
        self.repo.update("runs", run_id, {
            "completed_items": sum(item["status"] == WorkStatus.SUCCEEDED.value for item in items),
            "failed_items": sum(item["status"] == WorkStatus.FAILED.value for item in items),
            "input_tokens": sum(int(item.get("input_tokens", 0)) for item in extractions) + sum(int(item.get("input_tokens", 0)) for item in judges),
            "output_tokens": sum(int(item.get("output_tokens", 0)) for item in extractions) + sum(int(item.get("output_tokens", 0)) for item in judges),
            "estimated_cost_usd": sum(float(item.get("estimated_cost_usd", 0)) for item in extractions) + sum(float(item.get("estimated_cost_usd", 0)) for item in judges),
        })

    def _cancel_pending(self, run_id: str) -> None:
        for item in self.repo.find("work_items", {"run_id": run_id}):
            if item["status"] in {WorkStatus.PENDING.value, WorkStatus.LEASED.value}:
                self.repo.update("work_items", item["id"], {"status": WorkStatus.CANCELLED.value, "completed_at": utcnow()})

    def _finish(self, run_id: str) -> Run:
        self._refresh_progress(run_id)
        run = Run.model_validate(self.repo.get("runs", run_id))
        items = self.repo.find("work_items", {"run_id": run_id})
        if any(item["status"] in {WorkStatus.PENDING.value, WorkStatus.LEASED.value} for item in items):
            return run
        if run.status == RunStatus.CANCEL_REQUESTED:
            status = RunStatus.CANCELLED
        elif run.is_preflight:
            successful = next((item for item in items if item["status"] == WorkStatus.SUCCEEDED.value), None)
            extraction = None
            if successful:
                extraction = next(iter(self.repo.find("extractions", {"work_item_id": successful["id"]})), None)
            passed = bool(extraction and extraction.get("schema_valid"))
            result = PreflightResult(
                document_id=items[0]["document_id"], page_number=int(items[0]["page_number"]),
                extraction_id=extraction.get("id") if extraction else None, passed=passed,
                errors=[] if passed else ((extraction or {}).get("validation_errors") or ["preflight extraction failed"]),
            )
            status = RunStatus.PREFLIGHT_READY if passed else RunStatus.FAILED
            self.repo.update("runs", run_id, {"preflight_result": result.model_dump(mode="python")})
        elif any(item["status"] == WorkStatus.FAILED.value for item in items):
            status = RunStatus.COMPLETED_WITH_ERRORS
        else:
            status = RunStatus.COMPLETED
        self.repo.update("runs", run_id, {"status": status.value, "completed_at": utcnow()})
        finished = Run.model_validate(self.repo.get("runs", run_id))
        self.observer.record("run_completed", {
            "project_id": finished.project_id, "run_id": finished.id,
            "status": finished.status.value, "is_preflight": finished.is_preflight,
            "total_items": finished.total_items, "completed_items": finished.completed_items,
            "failed_items": finished.failed_items, "input_tokens": finished.input_tokens,
            "output_tokens": finished.output_tokens,
            "estimated_cost_usd": finished.estimated_cost_usd,
            "code_version": finished.snapshot.code_version,
        })
        return finished


def main() -> None:
    from .bootstrap import build_context

    run_id = os.getenv("HARNESS_RUN_ID")
    document_id = os.getenv("HARNESS_DOCUMENT_ID")
    if not run_id and not document_id:
        raise SystemExit("HARNESS_RUN_ID or HARNESS_DOCUMENT_ID is required")
    context = build_context()
    if document_id:
        document = context.service.render_document(document_id)
        print(f"document {document.id}: {document.render_status} ({document.page_count} pages)")
        return
    result = context.worker.process_run(run_id)
    print(f"run {result.id}: {result.status.value} ({result.completed_items}/{result.total_items})")


if __name__ == "__main__":
    main()
