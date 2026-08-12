"""Application service: uploads, version creation, labels, readiness, and run launch."""

from __future__ import annotations

import csv
import io
import json
import subprocess
from datetime import datetime
from typing import Any

from jsonschema import Draft202012Validator

from .llm import ExtractionClient
from .models import (
    Document, GroundTruth, ModelProfileVersion, PreflightResult, Project, ProjectMode,
    ProjectStatus, PromptVersion, Run, RunSnapshot, RunStatus, Stage, TaskSpecVersion,
    ToolProfileVersion, WorkItem, utcnow,
)
from .readiness import readiness_report
from .providers import ConnectorDefinition
from .repository import Repository
from .secrets import SecretResolver
from .storage import ArtifactStore, document_source_key, page_key, render_pdf, sha256_bytes
from .tools import validate_allowlist


def code_version() -> str:
    import os

    injected = os.getenv("HARNESS_CODE_VERSION")
    if injected:
        return injected
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
            timeout=3, check=True,
        )
        return result.stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


class HarnessService:
    def __init__(
        self, repo: Repository, artifacts: ArtifactStore, secrets: SecretResolver,
        llm: ExtractionClient, dispatcher_health=None, max_upload_bytes: int = 200 * 1024 * 1024,
        connectors: dict[str, ConnectorDefinition] | None = None,
    ) -> None:
        self.repo = repo
        self.artifacts = artifacts
        self.secrets = secrets
        self.llm = llm
        self.dispatcher_health = dispatcher_health
        self.max_upload_bytes = max_upload_bytes
        self.connectors = connectors or {}

    def create_project(self, name: str, mode: ProjectMode, description: str = "") -> Project:
        project = Project(name=name.strip(), mode=mode, description=description.strip())
        self.repo.put("projects", project)
        return project

    def project(self, project_id: str) -> Project:
        doc = self.repo.get("projects", project_id)
        if not doc:
            raise KeyError(f"project {project_id} not found")
        return Project.model_validate(doc)

    def save_project(self, project: Project) -> Project:
        project.updated_at = utcnow()
        self.repo.put("projects", project)
        return project

    def add_document(self, project_id: str, filename: str, data: bytes) -> Document:
        if len(data) > self.max_upload_bytes:
            raise ValueError(f"PDF exceeds the configured {self.max_upload_bytes // (1024 * 1024)} MB upload limit")
        if not filename.lower().endswith(".pdf") or not data.startswith(b"%PDF"):
            raise ValueError("only valid PDF uploads are supported")
        project = self.project(project_id)
        digest = sha256_bytes(data)
        existing = self.repo.find("documents", {"project_id": project_id, "sha256": digest})
        if existing:
            return Document.model_validate(existing[0])
        document = Document(
            project_id=project_id, name=filename, source_uri="pending", sha256=digest,
            size_bytes=len(data),
        )
        key = document_source_key(project_id, document.id, digest)
        document.source_uri = self.artifacts.put(key, data, "application/pdf")
        self.repo.put("documents", document)
        project.document_ids.append(document.id)
        if not project.preflight_document_id:
            project.preflight_document_id = document.id
        self.save_project(project)
        return document

    def render_document(self, document_id: str, dpi: int = 200, grayscale: bool = True) -> Document:
        raw = self.repo.get("documents", document_id)
        if not raw:
            raise KeyError(f"document {document_id} not found")
        document = Document.model_validate(raw)
        self.repo.update("documents", document.id, {"render_status": "running", "error": None})
        try:
            pages = render_pdf(self.artifacts.get(document.source_uri), grayscale=grayscale, dpi=dpi)
            uris = {
                str(index): self.artifacts.put(
                    page_key(document.project_id, document.id, document.sha256, index), page, "image/png"
                )
                for index, page in enumerate(pages, start=1)
            }
            document.page_count = len(pages)
            document.page_uris = uris
            document.render_status = "succeeded"
        except Exception as exc:
            document.render_status = "failed"
            document.error = f"{type(exc).__name__}: {exc}"
            self.repo.put("documents", document)
            raise
        self.repo.put("documents", document)
        return document

    def publish_task_spec(
        self, project_id: str, name: str, json_schema: dict[str, Any], field_rules: dict[str, Any] | None = None,
        aggregation_prompt_version_id: str | None = None,
    ) -> TaskSpecVersion:
        project = self.project(project_id)
        existing = self.repo.find("task_specs", {"project_id": project_id, "name": name})
        spec = TaskSpecVersion(
            project_id=project_id, version=len(existing) + 1, name=name, mode=project.mode,
            json_schema=json_schema, field_rules=field_rules or {},
            aggregation_enabled=aggregation_prompt_version_id is not None,
            aggregation_prompt_version_id=aggregation_prompt_version_id,
            tool_profile_version_id=project.tool_profile_version_id,
        )
        Draft202012Validator.check_schema(json_schema)
        self.repo.put("task_specs", spec)
        project.task_spec_version_id = spec.id
        self.save_project(project)
        return spec

    def publish_prompt(
        self, project_id: str, name: str, system_template: str, user_template: str,
        stage: str = "extraction",
    ) -> PromptVersion:
        existing = self.repo.find("prompt_versions", {"project_id": project_id, "name": name})
        prompt = PromptVersion(
            project_id=project_id, version=len(existing) + 1, name=name, stage=stage,
            system_template=system_template, user_template=user_template,
            allowed_variables=(
                ["document_name", "page_number"] if stage == "extraction"
                else ["document_name", "page_outputs"] if stage == "aggregation"
                else ["document_name", "page_number", "predicted", "expected", "deterministic_scores"]
            ),
        )
        self.repo.put("prompt_versions", prompt)
        if stage == "extraction":
            project = self.project(project_id)
            project.extraction_prompt_version_ids.append(prompt.id)
            self.save_project(project)
        return prompt

    def select_prompts(self, project_id: str, prompt_ids: list[str]) -> Project:
        available = {p["id"] for p in self.repo.find("prompt_versions", {"project_id": project_id}) if p.get("published") and p.get("stage") == "extraction"}
        if not prompt_ids or not set(prompt_ids) <= available:
            raise ValueError("select at least one published extraction prompt from this project")
        project = self.project(project_id)
        project.extraction_prompt_version_ids = prompt_ids
        return self.save_project(project)

    def publish_model(
        self, project_id: str, name: str, connector_id: str, model_id: str,
        request_parameters: dict[str, Any] | None = None,
        input_price_per_million: float | None = None,
        output_price_per_million: float | None = None,
    ) -> ModelProfileVersion:
        connector = self.connectors.get(connector_id)
        if not connector:
            raise ValueError("unknown administrator-approved connector")
        if model_id not in connector.allowed_models:
            raise ValueError(f"model {model_id!r} is not approved for connector {connector_id!r}")
        existing = self.repo.find("model_profiles", {"project_id": project_id, "name": name})
        profile = ModelProfileVersion(
            project_id=project_id, version=len(existing) + 1, name=name,
            connector_id=connector_id, base_url=connector.base_url,
            model_id=model_id, secret_ref=connector.secret_ref,
            request_parameters=request_parameters or {"temperature": 0},
            input_price_per_million=input_price_per_million,
            output_price_per_million=output_price_per_million,
        )
        self.repo.put("model_profiles", profile)
        project = self.project(project_id)
        project.model_profile_version_ids.append(profile.id)
        self.save_project(project)
        return profile

    def test_model(self, profile_id: str) -> tuple[bool, str]:
        raw = self.repo.get("model_profiles", profile_id)
        if not raw:
            raise KeyError(f"model profile {profile_id} not found")
        profile = ModelProfileVersion.model_validate(raw)
        ok, detail = self.llm.test_connection(profile, self.secrets.get(profile.secret_ref))
        self.repo.update("model_profiles", profile.id, {"tested_ok": ok, "tested_at": utcnow()})
        return ok, detail

    def select_models(self, project_id: str, profile_ids: list[str]) -> Project:
        available = {m["id"] for m in self.repo.find("model_profiles", {"project_id": project_id}) if m.get("published")}
        if not profile_ids or not set(profile_ids) <= available:
            raise ValueError("select at least one published model profile from this project")
        project = self.project(project_id)
        project.model_profile_version_ids = profile_ids
        return self.save_project(project)

    def publish_tool_profile(self, project_id: str, name: str, allowed_tools: list[str]) -> ToolProfileVersion:
        errors = validate_allowlist(allowed_tools, "extraction")
        if errors:
            raise ValueError("; ".join(errors))
        if "validate_schema" not in allowed_tools:
            raise ValueError("extraction tool profiles must include validate_schema")
        existing = self.repo.find("tool_profiles", {"project_id": project_id, "name": name})
        profile = ToolProfileVersion(
            project_id=project_id, version=len(existing) + 1, name=name,
            allowed_tools=allowed_tools, tested_ok=True, tested_at=utcnow(),
        )
        self.repo.put("tool_profiles", profile)
        project = self.project(project_id)
        project.tool_profile_version_id = profile.id
        self.save_project(project)
        return profile

    def import_ground_truth(self, project_id: str, payload: bytes, filename: str) -> list[GroundTruth]:
        project = self.project(project_id)
        if project.mode != ProjectMode.BENCHMARK:
            raise ValueError("ground truth can only be imported into benchmark projects")
        if filename.lower().endswith(".json"):
            parsed = json.loads(payload)
            rows = parsed if isinstance(parsed, list) else parsed.get("rows", [])
        elif filename.lower().endswith(".csv"):
            rows = list(csv.DictReader(io.StringIO(payload.decode("utf-8-sig"))))
        else:
            raise ValueError("ground truth must be CSV or JSON")
        task_raw = self.repo.get("task_specs", project.task_spec_version_id or "")
        if not task_raw:
            raise ValueError("publish a task schema before importing ground truth")
        spec = TaskSpecVersion.model_validate(task_raw)
        validator = Draft202012Validator(spec.json_schema)
        imported: list[GroundTruth] = []
        for row in rows:
            document_id = str(row.get("document_id", ""))
            page_raw = row.get("page_number")
            page_number = int(page_raw) if page_raw not in (None, "") else None
            values = row.get("values") if isinstance(row.get("values"), dict) else {
                key: _coerce_import_value(row.get(key), schema)
                for key, schema in spec.json_schema.get("properties", {}).items()
                if key in row
            }
            errors = [f"{'.'.join(map(str, e.path)) or '$'}: {e.message}" for e in validator.iter_errors(values)]
            document = self.repo.get("documents", document_id)
            if not document or document_id not in project.document_ids:
                errors.append("document_id is not part of this project")
            elif page_number is not None and page_number > int(document.get("page_count") or 0):
                errors.append("page_number is outside the rendered document")
            previous = self.repo.find("ground_truth", {"project_id": project_id, "document_id": document_id, "page_number": page_number})
            label = GroundTruth(
                project_id=project_id, document_id=document_id, page_number=page_number,
                values=values, revision=max((int(v["revision"]) for v in previous), default=0) + 1,
                approved=False, import_source=filename, validation_errors=errors,
            )
            self.repo.put("ground_truth", label)
            imported.append(label)
        return imported

    def revise_ground_truth(
        self, label_id: str, values: dict[str, Any], approved: bool,
    ) -> GroundTruth:
        raw = self.repo.get("ground_truth", label_id)
        if not raw:
            raise KeyError(f"ground-truth label {label_id} not found")
        current = GroundTruth.model_validate(raw)
        project = self.project(current.project_id)
        task_raw = self.repo.get("task_specs", project.task_spec_version_id or "")
        if not task_raw:
            raise ValueError("project has no task schema")
        spec = TaskSpecVersion.model_validate(task_raw)
        errors = [
            f"{'.'.join(map(str, e.path)) or '$'}: {e.message}"
            for e in Draft202012Validator(spec.json_schema).iter_errors(values)
        ]
        if approved and errors:
            raise ValueError("invalid ground truth cannot be approved: " + "; ".join(errors))
        revision = GroundTruth(
            project_id=current.project_id, document_id=current.document_id,
            page_number=current.page_number, values=values, revision=current.revision + 1,
            approved=approved, import_source=current.import_source,
            validation_errors=errors,
        )
        self.repo.put("ground_truth", revision)
        return revision

    def readiness(self, project_id: str):
        report = readiness_report(self.project(project_id), self.repo, self.artifacts, self.dispatcher_health)
        self.repo.update("projects", project_id, {"status": ProjectStatus.READY.value if report.ready else ProjectStatus.DRAFT.value})
        return report

    def create_run(
        self, project_id: str, preflight: bool, llm_judge_enabled: bool = False,
        judge_model_profile_version_id: str | None = None,
        judge_prompt_version_id: str | None = None,
    ) -> Run:
        project = self.project(project_id)
        report = self.readiness(project_id)
        if not report.ready:
            failed = [c.label for c in report.checks if c.blocking and not c.passed]
            raise ValueError(f"project is not ready: {', '.join(failed)}")
        all_labels = self.repo.find("ground_truth", {"project_id": project_id})
        latest_labels: dict[tuple[str, int | None], dict[str, Any]] = {}
        for label in all_labels:
            key = (label["document_id"], label.get("page_number"))
            if key not in latest_labels or int(label["revision"]) > int(latest_labels[key]["revision"]):
                latest_labels[key] = label
        labels = [label for label in latest_labels.values() if label.get("approved")]
        if llm_judge_enabled:
            judge_model = self.repo.get("model_profiles", judge_model_profile_version_id or "")
            judge_prompt = self.repo.get("prompt_versions", judge_prompt_version_id or "")
            if not judge_model or not judge_model.get("tested_ok"):
                raise ValueError("optional LLM judge requires a tested judge model profile")
            if not judge_prompt or judge_prompt.get("stage") != "judge" or not judge_prompt.get("published"):
                raise ValueError("optional LLM judge requires a published judge prompt")
        snapshot = RunSnapshot(
            project_id=project_id, mode=project.mode,
            task_spec_version_id=project.task_spec_version_id or "",
            prompt_version_ids=project.extraction_prompt_version_ids,
            model_profile_version_ids=project.model_profile_version_ids,
            tool_profile_version_id=project.tool_profile_version_id,
            document_ids=project.document_ids,
            ground_truth_revisions={label["id"]: int(label["revision"]) for label in labels},
            ground_truth_ids={
                f"{label['document_id']}:{label.get('page_number') or 'document'}": label["id"]
                for label in labels
            },
            code_version=code_version(), llm_judge_enabled=llm_judge_enabled,
            judge_model_profile_version_id=judge_model_profile_version_id,
            judge_prompt_version_id=judge_prompt_version_id,
        )
        run = Run(
            project_id=project_id, snapshot=snapshot, is_preflight=preflight,
            status=RunStatus.PREFLIGHT_PENDING if preflight else RunStatus.PREFLIGHT_READY,
        )
        self.repo.put("runs", run)
        work = self._work_items(run, project)
        run.total_items = len(work)
        self.repo.put("runs", run)
        for item in work:
            self.repo.put("work_items", item)
        return run

    def _work_items(self, run: Run, project: Project) -> list[WorkItem]:
        items: list[WorkItem] = []
        documents = run.snapshot.document_ids
        models = run.snapshot.model_profile_version_ids
        prompts = run.snapshot.prompt_version_ids
        if run.is_preflight:
            documents = [project.preflight_document_id or documents[0]]
            models = models[:1]
            prompts = prompts[:1]
        for document_id in documents:
            raw = self.repo.get("documents", document_id)
            if not raw:
                continue
            pages = [project.preflight_page_number] if run.is_preflight else sorted(int(p) for p in raw.get("page_uris", {}))
            for page in pages:
                for model_id in models:
                    for prompt_id in prompts:
                        items.append(WorkItem(
                            run_id=run.id, project_id=run.project_id, stage=Stage.EXTRACT,
                            document_id=document_id, page_number=page,
                            model_profile_version_id=model_id, prompt_version_id=prompt_id,
                            source_prompt_version_id=prompt_id,
                        ))
        return items

    def approve_preflight(
        self, preflight_run_id: str, llm_judge_enabled: bool = False,
        judge_model_profile_version_id: str | None = None,
        judge_prompt_version_id: str | None = None,
    ) -> Run:
        raw = self.repo.get("runs", preflight_run_id)
        if not raw:
            raise KeyError(f"run {preflight_run_id} not found")
        preflight = Run.model_validate(raw)
        if not preflight.is_preflight or not preflight.preflight_result or not preflight.preflight_result.passed:
            raise ValueError("a successful preflight is required before full launch")
        existing = self.repo.find("runs", {"approved_preflight_run_id": preflight_run_id})
        if existing:
            return Run.model_validate(existing[0])
        if llm_judge_enabled:
            judge_model = self.repo.get("model_profiles", judge_model_profile_version_id or "")
            judge_prompt = self.repo.get("prompt_versions", judge_prompt_version_id or "")
            if not judge_model or not judge_model.get("tested_ok"):
                raise ValueError("optional LLM judge requires a tested judge model profile")
            if not judge_prompt or judge_prompt.get("stage") != "judge":
                raise ValueError("optional LLM judge requires a published judge prompt")
        snapshot = preflight.snapshot.model_copy(update={
            "llm_judge_enabled": llm_judge_enabled,
            "judge_model_profile_version_id": judge_model_profile_version_id,
            "judge_prompt_version_id": judge_prompt_version_id,
        })
        batch = Run(
            project_id=preflight.project_id, snapshot=snapshot, is_preflight=False,
            status=RunStatus.PREFLIGHT_READY, approved_preflight_run_id=preflight_run_id,
        )
        project = self.project(preflight.project_id)
        work = self._work_items(batch, project)
        batch.total_items = len(work)
        self.repo.put("runs", batch)
        for item in work:
            self.repo.put("work_items", item)
        return batch


def _coerce_import_value(value: Any, schema: dict[str, Any]) -> Any:
    if value is None:
        return None
    kind = schema.get("type")
    if kind == "array" and isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return [v.strip() for v in value.split("|") if v.strip()]
    if kind == "integer":
        return int(value)
    if kind == "number":
        return float(value)
    if kind == "boolean" and isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return value
