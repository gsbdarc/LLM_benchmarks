"""One readiness policy used by both Streamlit and the dispatcher."""

from __future__ import annotations

from typing import Callable

from .models import Project, ProjectMode, ReadinessCheck, ReadinessReport
from .repository import Repository
from .storage import ArtifactStore
from .tools import validate_allowlist


def readiness_report(
    project: Project,
    repo: Repository,
    artifacts: ArtifactStore,
    dispatcher_health: Callable[[], tuple[bool, str]] | None = None,
) -> ReadinessReport:
    checks: list[ReadinessCheck] = []

    def add(key: str, label: str, result: tuple[bool, str]) -> None:
        checks.append(ReadinessCheck(key=key, label=label, passed=result[0], detail=result[1]))

    add("mongo", "MongoDB", repo.health())
    add("artifacts", "Artifact storage", artifacts.health())
    add("dispatcher", "Job dispatcher", dispatcher_health() if dispatcher_health else (True, "local worker mode"))

    documents = [repo.get("documents", item_id) for item_id in project.document_ids]
    rendered = [d for d in documents if d and d.get("render_status") == "succeeded" and d.get("page_uris")]
    add("documents", "Rendered documents", (
        bool(documents) and len(rendered) == len(documents) == len(project.document_ids),
        f"{len(rendered)} of {len(project.document_ids)} document(s) rendered",
    ))

    task = repo.get("task_specs", project.task_spec_version_id or "")
    add("task_spec", "Published task schema", (
        bool(task and task.get("published")), "task schema is published" if task else "no task schema selected"
    ))

    prompts = [repo.get("prompt_versions", item_id) for item_id in project.extraction_prompt_version_ids]
    valid_prompts = [p for p in prompts if p and p.get("published") and p.get("stage") == "extraction"]
    add("prompts", "Extraction prompts", (
        bool(valid_prompts) and len(valid_prompts) == len(prompts),
        f"{len(valid_prompts)} published extraction prompt(s)",
    ))

    models = [repo.get("model_profiles", item_id) for item_id in project.model_profile_version_ids]
    tested_models = [m for m in models if m and m.get("published") and m.get("tested_ok") and m.get("supports_vision")]
    add("models", "Tested vision models", (
        bool(tested_models) and len(tested_models) == len(models),
        f"{len(tested_models)} of {len(models)} selected model(s) passed connection tests",
    ))

    profile = repo.get("tool_profiles", project.tool_profile_version_id or "")
    tool_errors = validate_allowlist(profile.get("allowed_tools", []), "extraction") if profile else []
    add("tools", "Built-in MCP tools", (
        bool(profile and profile.get("published") and profile.get("tested_ok") and not tool_errors),
        "; ".join(tool_errors) if tool_errors else ("tool profile is healthy" if profile else "no tool profile selected"),
    ))

    if project.mode == ProjectMode.BENCHMARK:
        labels = repo.find("ground_truth", {"project_id": project.id})
        latest: dict[tuple[str, int | None], dict] = {}
        for label in labels:
            key = (label["document_id"], label.get("page_number"))
            if key not in latest or int(label["revision"]) > int(latest[key]["revision"]):
                latest[key] = label
        approved = {
            key for key, label in latest.items()
            if label.get("approved") and not label.get("validation_errors")
        }
        missing = []
        for document in rendered:
            document_id = document["id"]
            if (document_id, None) in approved:
                continue
            for page in range(1, int(document.get("page_count") or 0) + 1):
                if (document_id, page) not in approved:
                    missing.append(f"{document_id}:page:{page}")
        add("ground_truth", "Approved ground truth", (
            not missing, "all selected pages have approved labels" if not missing else f"missing approved labels for {len(missing)} page(s)",
        ))

    return ReadinessReport(project_id=project.id, ready=all(c.passed or not c.blocking for c in checks), checks=checks)
