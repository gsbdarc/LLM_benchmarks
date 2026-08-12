"""Stable, versioned domain contracts shared by the UI, dispatcher, and worker."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from hashlib import sha256
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return str(uuid4())


def content_hash(value: Any) -> str:
    import json

    blob = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return sha256(blob.encode()).hexdigest()


class HarnessModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class ProjectMode(str, Enum):
    EXTRACTION = "extraction"
    BENCHMARK = "benchmark"


class ProjectStatus(str, Enum):
    DRAFT = "draft"
    READY = "ready"
    ARCHIVED = "archived"


class RunStatus(str, Enum):
    DRAFT = "draft"
    PREFLIGHT_PENDING = "preflight_pending"
    PREFLIGHT_RUNNING = "preflight_running"
    PREFLIGHT_READY = "preflight_ready"
    DISPATCHING = "dispatching"
    DISPATCH_UNKNOWN = "dispatch_unknown"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    COMPLETED_WITH_ERRORS = "completed_with_errors"
    FAILED = "failed"
    CANCEL_REQUESTED = "cancel_requested"
    CANCELLED = "cancelled"


class WorkStatus(str, Enum):
    PENDING = "pending"
    LEASED = "leased"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Stage(str, Enum):
    RENDER = "render"
    EXTRACT = "extract"
    AGGREGATE = "aggregate"
    SCORE = "score"
    JUDGE = "judge"


class FieldRule(HarnessModel):
    description: str = ""
    required: bool = True
    normalization: Literal["none", "trim", "lower", "date"] = "trim"
    scoring: Literal[
        "auto", "exact", "normalized_exact", "text_similarity", "numeric",
        "date", "boolean", "list",
    ] = "auto"
    numeric_tolerance: float | None = Field(default=None, ge=0)


class TaskSpecVersion(HarnessModel):
    id: str = Field(default_factory=new_id)
    project_id: str
    version: int = Field(ge=1)
    name: str
    mode: ProjectMode
    json_schema: dict[str, Any]
    field_rules: dict[str, FieldRule] = Field(default_factory=dict)
    aggregation_enabled: bool = False
    aggregation_prompt_version_id: str | None = None
    tool_profile_version_id: str | None = None
    published: bool = True
    content_hash: str = ""
    created_at: datetime = Field(default_factory=utcnow)
    created_by: str = "prototype-user"

    @model_validator(mode="after")
    def validate_contract(self) -> "TaskSpecVersion":
        if self.json_schema.get("type") != "object":
            raise ValueError("task json_schema must have top-level type 'object'")
        properties = self.json_schema.get("properties")
        if not isinstance(properties, dict) or not properties:
            raise ValueError("task json_schema requires at least one property")
        unknown = set(self.field_rules) - set(properties)
        if unknown:
            raise ValueError(f"field_rules reference unknown schema fields: {sorted(unknown)}")
        if self.aggregation_enabled and not self.aggregation_prompt_version_id:
            raise ValueError("aggregation_enabled requires aggregation_prompt_version_id")
        if not self.content_hash:
            self.content_hash = content_hash({
                "schema": self.json_schema,
                "rules": {k: v.model_dump(mode="json") for k, v in self.field_rules.items()},
                "aggregation": self.aggregation_enabled,
                "aggregation_prompt": self.aggregation_prompt_version_id,
                "tools": self.tool_profile_version_id,
            })
        return self


class PromptVersion(HarnessModel):
    id: str = Field(default_factory=new_id)
    project_id: str
    version: int = Field(ge=1)
    name: str
    stage: Literal["extraction", "aggregation", "judge"] = "extraction"
    system_template: str
    user_template: str
    allowed_variables: list[str] = Field(default_factory=lambda: ["document_name", "page_number"])
    published: bool = True
    content_hash: str = ""
    created_at: datetime = Field(default_factory=utcnow)
    created_by: str = "prototype-user"

    @model_validator(mode="after")
    def set_hash(self) -> "PromptVersion":
        import string

        used = {field for _, field, _, _ in string.Formatter().parse(self.user_template) if field}
        unsupported = used - set(self.allowed_variables)
        if unsupported:
            raise ValueError(f"unsupported prompt variables: {sorted(unsupported)}")
        if not self.content_hash:
            self.content_hash = content_hash({
                "stage": self.stage,
                "system": self.system_template,
                "user": self.user_template,
            })
        return self


class ModelProfileVersion(HarnessModel):
    id: str = Field(default_factory=new_id)
    project_id: str
    version: int = Field(ge=1)
    name: str
    connector_id: str
    base_url: str
    model_id: str
    secret_ref: str
    request_parameters: dict[str, Any] = Field(default_factory=lambda: {"temperature": 0})
    supports_vision: bool = True
    input_price_per_million: float | None = Field(default=None, ge=0)
    output_price_per_million: float | None = Field(default=None, ge=0)
    tested_ok: bool = False
    tested_at: datetime | None = None
    published: bool = True
    content_hash: str = ""
    created_at: datetime = Field(default_factory=utcnow)

    @model_validator(mode="after")
    def set_hash(self) -> "ModelProfileVersion":
        if not self.content_hash:
            self.content_hash = content_hash({
                "base_url": self.base_url,
                "model": self.model_id,
                "params": self.request_parameters,
                "secret_ref": self.secret_ref,
            })
        return self


class ToolProfileVersion(HarnessModel):
    id: str = Field(default_factory=new_id)
    project_id: str
    version: int = Field(ge=1)
    name: str
    allowed_tools: list[str] = Field(default_factory=list)
    configuration: dict[str, Any] = Field(default_factory=dict)
    tested_ok: bool = False
    tested_at: datetime | None = None
    published: bool = True
    content_hash: str = ""
    created_at: datetime = Field(default_factory=utcnow)

    @model_validator(mode="after")
    def set_hash(self) -> "ToolProfileVersion":
        if not self.content_hash:
            self.content_hash = content_hash({"tools": sorted(self.allowed_tools), "config": self.configuration})
        return self


class Project(HarnessModel):
    id: str = Field(default_factory=new_id)
    name: str
    description: str = ""
    mode: ProjectMode = ProjectMode.EXTRACTION
    status: ProjectStatus = ProjectStatus.DRAFT
    document_ids: list[str] = Field(default_factory=list)
    task_spec_version_id: str | None = None
    extraction_prompt_version_ids: list[str] = Field(default_factory=list)
    model_profile_version_ids: list[str] = Field(default_factory=list)
    tool_profile_version_id: str | None = None
    preflight_document_id: str | None = None
    preflight_page_number: int = Field(default=1, ge=1)
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    created_by: str = "prototype-user"


class Document(HarnessModel):
    id: str = Field(default_factory=new_id)
    project_id: str
    name: str
    source_uri: str
    sha256: str
    size_bytes: int = Field(ge=0)
    content_type: str = "application/pdf"
    page_count: int | None = Field(default=None, ge=1)
    render_status: Literal[
        "pending", "dispatching", "queued", "running", "succeeded", "failed", "dispatch_unknown",
    ] = "pending"
    page_uris: dict[str, str] = Field(default_factory=dict)
    render_execution_name: str | None = None
    error: str | None = None
    created_at: datetime = Field(default_factory=utcnow)


class GroundTruth(HarnessModel):
    id: str = Field(default_factory=new_id)
    project_id: str
    document_id: str
    page_number: int | None = Field(default=None, ge=1)
    values: dict[str, Any]
    revision: int = Field(default=1, ge=1)
    approved: bool = False
    import_source: str | None = None
    validation_errors: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utcnow)


class RunSnapshot(HarnessModel):
    project_id: str
    mode: ProjectMode
    task_spec_version_id: str
    prompt_version_ids: list[str]
    model_profile_version_ids: list[str]
    tool_profile_version_id: str | None = None
    document_ids: list[str]
    ground_truth_revisions: dict[str, int] = Field(default_factory=dict)
    ground_truth_ids: dict[str, str] = Field(default_factory=dict)
    code_version: str = "unknown"
    llm_judge_enabled: bool = False
    judge_model_profile_version_id: str | None = None
    judge_prompt_version_id: str | None = None


class PreflightResult(HarnessModel):
    document_id: str
    page_number: int
    extraction_id: str | None = None
    passed: bool
    errors: list[str] = Field(default_factory=list)
    completed_at: datetime = Field(default_factory=utcnow)


class Run(HarnessModel):
    id: str = Field(default_factory=new_id)
    project_id: str
    snapshot: RunSnapshot
    status: RunStatus = RunStatus.DRAFT
    is_preflight: bool = False
    preflight_result: PreflightResult | None = None
    approved_at: datetime | None = None
    cloud_execution_name: str | None = None
    dispatch_started_at: datetime | None = None
    approved_preflight_run_id: str | None = None
    total_items: int = 0
    completed_items: int = 0
    failed_items: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0
    error: str | None = None
    created_at: datetime = Field(default_factory=utcnow)
    started_at: datetime | None = None
    completed_at: datetime | None = None


class WorkItem(HarnessModel):
    id: str = Field(default_factory=new_id)
    run_id: str
    project_id: str
    stage: Stage
    document_id: str
    page_number: int | None = Field(default=None, ge=1)
    model_profile_version_id: str | None = None
    prompt_version_id: str | None = None
    source_prompt_version_id: str | None = None
    status: WorkStatus = WorkStatus.PENDING
    attempt: int = 0
    lease_owner: str | None = None
    lease_expires_at: datetime | None = None
    error: str | None = None
    created_at: datetime = Field(default_factory=utcnow)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    identity: str = ""

    @model_validator(mode="after")
    def set_identity(self) -> "WorkItem":
        if not self.identity:
            self.identity = content_hash({
                "run": self.run_id,
                "stage": self.stage.value,
                "document": self.document_id,
                "page": self.page_number,
                "model": self.model_profile_version_id,
                "prompt": self.prompt_version_id,
                "source_prompt": self.source_prompt_version_id,
            })
        return self


class Extraction(HarnessModel):
    id: str = Field(default_factory=new_id)
    work_item_id: str
    run_id: str
    project_id: str
    document_id: str
    page_number: int | None = None
    model_profile_version_id: str
    prompt_version_id: str
    source_prompt_version_id: str | None = None
    output: dict[str, Any] | None = None
    raw_response: str | None = None
    repair_response: str | None = None
    schema_valid: bool = False
    validation_errors: list[str] = Field(default_factory=list)
    repair_attempts: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0
    latency_seconds: float = 0
    created_at: datetime = Field(default_factory=utcnow)


class Evaluation(HarnessModel):
    id: str = Field(default_factory=new_id)
    run_id: str
    extraction_id: str
    deterministic: dict[str, Any]
    llm_judge: dict[str, Any] | None = None
    created_at: datetime = Field(default_factory=utcnow)


class TraceEvent(HarnessModel):
    id: str = Field(default_factory=new_id)
    run_id: str
    work_item_id: str
    sequence: int = Field(ge=0)
    attempt: int = Field(default=1, ge=1)
    event_type: Literal["model_request", "model_response", "tool_call", "tool_result", "retry", "error"]
    name: str
    payload: dict[str, Any] = Field(default_factory=dict)
    duration_seconds: float | None = None
    created_at: datetime = Field(default_factory=utcnow)


class ReadinessCheck(HarnessModel):
    key: str
    label: str
    passed: bool
    detail: str
    blocking: bool = True


class ReadinessReport(HarnessModel):
    project_id: str
    ready: bool
    checks: list[ReadinessCheck]
    generated_at: datetime = Field(default_factory=utcnow)
