from __future__ import annotations

from pathlib import Path

from pdf_harness.llm import LLMResult
from pdf_harness.models import ProjectMode, RunStatus
from pdf_harness.providers import ConnectorDefinition
from pdf_harness.repository import InMemoryRepository
from pdf_harness.service import HarnessService
from pdf_harness.storage import LocalArtifactStore, page_key
from pdf_harness.worker import HarnessWorker


SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}},
    "required": ["name"],
    "additionalProperties": False,
}


class FakeSecrets:
    def get(self, reference):
        assert reference == "env:STANFORD_API_KEY"
        return "secret-value"


class FakeLLM:
    def test_connection(self, profile, api_key):
        return api_key == "secret-value", "fake endpoint healthy"

    def extract(self, profile, prompt, api_key, image, image_content_type, schema, variables):
        assert api_key == "secret-value"
        assert image == b"png"
        return LLMResult(
            output={"name": "Alice"}, raw_response='{"name":"Alice"}',
            input_tokens=10, output_tokens=4, latency_seconds=.25,
        )

    def aggregate(self, profile, prompt, api_key, page_outputs, schema, variables):
        assert page_outputs
        return LLMResult(
            output={"name": "Alice"}, raw_response='{"name":"Alice"}',
            input_tokens=12, output_tokens=4, latency_seconds=.3,
        )

    def judge(self, profile, prompt, api_key, predicted, expected, deterministic_scores, variables):
        return LLMResult(
            output={"score": 1.0, "rationale": "Matches", "needs_review": False},
            raw_response='{"score":1,"rationale":"Matches","needs_review":false}',
            input_tokens=20, output_tokens=8, latency_seconds=.2,
        )


class RepairingLLM(FakeLLM):
    def extract(self, *args, **kwargs):
        return LLMResult(
            output={"wrong": "shape"}, raw_response='{"wrong":"shape"}',
            input_tokens=5, output_tokens=3, latency_seconds=.1,
        )

    def repair(self, profile, api_key, raw_response, schema, variables):
        return LLMResult(
            output={"name": "Alice"}, raw_response='{"name":"Alice"}',
            input_tokens=4, output_tokens=2, latency_seconds=.1,
        )


def build(tmp_path: Path):
    repo = InMemoryRepository()
    artifacts = LocalArtifactStore(tmp_path)
    llm = FakeLLM()
    secrets = FakeSecrets()
    connectors = {"fake": ConnectorDefinition(
        id="fake", label="Fake", base_url="https://example.test/v1",
        secret_ref="env:STANFORD_API_KEY", allowed_models=["fake-model"],
    )}
    service = HarnessService(repo, artifacts, secrets, llm, connectors=connectors)
    worker = HarnessWorker(repo, artifacts, secrets, llm, concurrency=2)
    return repo, artifacts, service, worker


def configured_project(tmp_path: Path):
    repo, artifacts, service, worker = build(tmp_path)
    project = service.create_project("Demo", ProjectMode.EXTRACTION)
    document = service.add_document(project.id, "sample.pdf", b"%PDF fake")
    uri = artifacts.put(page_key(project.id, document.id, document.sha256, 1), b"png", "image/png")
    repo.update("documents", document.id, {"render_status": "succeeded", "page_count": 1, "page_uris": {"1": uri}})
    service.publish_task_spec(project.id, "people", SCHEMA)
    service.publish_prompt(
        project.id, "extract_v1", "Return JSON.",
        "Extract from {document_name}, page {page_number}.",
    )
    model = service.publish_model(
        project.id, "fake", "fake", "fake-model",
        input_price_per_million=1, output_price_per_million=2,
    )
    assert service.test_model(model.id)[0]
    service.publish_tool_profile(project.id, "safe", ["validate_schema", "normalize_text"])
    return repo, service, worker, project


def test_end_to_end_preflight_approval_and_batch(tmp_path):
    repo, service, worker, project = configured_project(tmp_path)
    assert service.readiness(project.id).ready

    preflight = service.create_run(project.id, preflight=True)
    assert repo.queue_run(preflight.id)
    finished = worker.process_run(preflight.id)
    assert finished.status == RunStatus.PREFLIGHT_READY
    assert finished.preflight_result.passed
    extraction = repo.find("extractions", {"run_id": preflight.id})[0]
    assert extraction["schema_valid"]
    assert extraction["estimated_cost_usd"] > 0
    assert len(repo.find("traces", {"work_item_id": extraction["work_item_id"]})) == 4

    batch = service.approve_preflight(preflight.id)
    assert batch.status == RunStatus.PREFLIGHT_READY
    assert repo.queue_run(batch.id)
    completed = worker.process_run(batch.id)
    assert completed.status == RunStatus.COMPLETED
    assert completed.completed_items == completed.total_items == 1


def test_duplicate_pdf_upload_reuses_document(tmp_path):
    repo, artifacts, service, worker = build(tmp_path)
    project = service.create_project("Duplicates", ProjectMode.EXTRACTION)
    first = service.add_document(project.id, "first.pdf", b"%PDF same")
    second = service.add_document(project.id, "renamed.pdf", b"%PDF same")
    assert second.id == first.id
    assert len(repo.find("documents", {"project_id": project.id})) == 1
    assert service.project(project.id).document_ids == [first.id]


def test_preflight_approval_is_snapshot_bound_and_idempotent(tmp_path):
    repo, service, worker, project = configured_project(tmp_path)
    preflight = service.create_run(project.id, preflight=True)
    repo.queue_run(preflight.id)
    worker.process_run(preflight.id)
    original_prompts = list(preflight.snapshot.prompt_version_ids)
    service.publish_prompt(project.id, "new", "New system", "Read {document_name} page {page_number}")
    first = service.approve_preflight(preflight.id)
    second = service.approve_preflight(preflight.id)
    assert first.id == second.id
    assert first.snapshot.prompt_version_ids == original_prompts
    assert first.total_items == 1


def test_benchmark_readiness_requires_approved_labels(tmp_path):
    repo, artifacts, service, worker = build(tmp_path)
    project = service.create_project("Benchmark", ProjectMode.BENCHMARK)
    document = service.add_document(project.id, "sample.pdf", b"%PDF fake")
    uri = artifacts.put(page_key(project.id, document.id, document.sha256, 1), b"png", "image/png")
    repo.update("documents", document.id, {"render_status": "succeeded", "page_count": 1, "page_uris": {"1": uri}})
    service.publish_task_spec(project.id, "people", SCHEMA)
    service.publish_prompt(project.id, "extract", "Return JSON", "Read {document_name} page {page_number}")
    model = service.publish_model(project.id, "fake", "fake", "fake-model")
    service.test_model(model.id)
    service.publish_tool_profile(project.id, "safe", ["validate_schema"])
    report = service.readiness(project.id)
    assert not report.ready
    assert next(c for c in report.checks if c.key == "ground_truth").passed is False

    imported = service.import_ground_truth(
        project.id,
        f'document_id,page_number,name\n{document.id},1,Alice\n'.encode(),
        "truth.csv",
    )
    assert imported[0].approved is False
    approved = service.revise_ground_truth(imported[0].id, {"name": "Alice"}, approved=True)
    assert approved.revision == 2
    assert service.readiness(project.id).ready


def test_benchmark_scores_snapshotted_label_revision(tmp_path):
    repo, artifacts, service, worker = build(tmp_path)
    project = service.create_project("Benchmark", ProjectMode.BENCHMARK)
    document = service.add_document(project.id, "sample.pdf", b"%PDF fake")
    uri = artifacts.put(page_key(project.id, document.id, document.sha256, 1), b"png", "image/png")
    repo.update("documents", document.id, {"render_status": "succeeded", "page_count": 1, "page_uris": {"1": uri}})
    service.publish_task_spec(project.id, "people", SCHEMA)
    service.publish_prompt(project.id, "extract", "Return JSON", "Read {document_name} page {page_number}")
    model = service.publish_model(project.id, "fake", "fake", "fake-model")
    service.test_model(model.id)
    service.publish_tool_profile(project.id, "safe", ["validate_schema"])
    imported = service.import_ground_truth(
        project.id, f'document_id,page_number,name\n{document.id},1,Alice\n'.encode(), "truth.csv",
    )[0]
    approved = service.revise_ground_truth(imported.id, {"name": "Alice"}, approved=True)
    run = service.create_run(project.id, preflight=True)
    service.revise_ground_truth(approved.id, {"name": "Bob"}, approved=True)
    repo.queue_run(run.id)
    worker.process_run(run.id)
    evaluation = repo.find("evaluations", {"run_id": run.id})[0]
    assert evaluation["deterministic"]["overall_score"] == 1.0


def test_invalid_structured_output_gets_one_bounded_repair(tmp_path):
    repo, artifacts, service, _ = build(tmp_path)
    llm = RepairingLLM()
    service.llm = llm
    worker = HarnessWorker(repo, artifacts, service.secrets, llm)
    project = service.create_project("Repair", ProjectMode.EXTRACTION)
    document = service.add_document(project.id, "sample.pdf", b"%PDF fake")
    uri = artifacts.put(page_key(project.id, document.id, document.sha256, 1), b"png", "image/png")
    repo.update("documents", document.id, {"render_status": "succeeded", "page_count": 1, "page_uris": {"1": uri}})
    service.publish_task_spec(project.id, "people", SCHEMA)
    service.publish_prompt(project.id, "extract", "Return JSON", "Read {document_name} page {page_number}")
    model = service.publish_model(project.id, "fake", "fake", "fake-model")
    service.test_model(model.id)
    service.publish_tool_profile(project.id, "safe", ["validate_schema"])
    run = service.create_run(project.id, preflight=True)
    repo.queue_run(run.id)
    worker.process_run(run.id)
    extraction = repo.find("extractions", {"run_id": run.id})[0]
    assert extraction["schema_valid"] is True
    assert extraction["repair_attempts"] == 1
    assert extraction["raw_response"] == '{"wrong":"shape"}'
    assert extraction["repair_response"] == '{"name":"Alice"}'


def test_optional_document_aggregation_runs_after_page_extraction(tmp_path):
    repo, artifacts, service, worker = build(tmp_path)
    project = service.create_project("Aggregate", ProjectMode.EXTRACTION)
    document = service.add_document(project.id, "sample.pdf", b"%PDF fake")
    uris = {
        str(page): artifacts.put(page_key(project.id, document.id, document.sha256, page), b"png", "image/png")
        for page in (1, 2)
    }
    repo.update("documents", document.id, {"render_status": "succeeded", "page_count": 2, "page_uris": uris})
    aggregation = service.publish_prompt(
        project.id, "combine", "Combine pages", "Combine {document_name}: {page_outputs}", stage="aggregation",
    )
    service.publish_task_spec(project.id, "people", SCHEMA, aggregation_prompt_version_id=aggregation.id)
    service.publish_prompt(project.id, "extract", "Return JSON", "Read {document_name} page {page_number}")
    model = service.publish_model(project.id, "fake", "fake", "fake-model")
    service.test_model(model.id)
    service.publish_tool_profile(project.id, "safe", ["validate_schema"])
    preflight = service.create_run(project.id, preflight=True)
    repo.queue_run(preflight.id)
    worker.process_run(preflight.id)
    batch = service.approve_preflight(preflight.id)
    repo.queue_run(batch.id)
    completed = worker.process_run(batch.id)
    assert completed.status == RunStatus.COMPLETED
    results = repo.find("extractions", {"run_id": batch.id})
    assert len([item for item in results if item["page_number"] is not None]) == 2
    aggregate = [item for item in results if item["page_number"] is None]
    assert len(aggregate) == 1
    assert aggregate[0]["source_prompt_version_id"] == batch.snapshot.prompt_version_ids[0]


def test_optional_llm_judge_is_separate_from_deterministic_score(tmp_path):
    repo, artifacts, service, worker = build(tmp_path)
    project = service.create_project("Judged", ProjectMode.BENCHMARK)
    document = service.add_document(project.id, "sample.pdf", b"%PDF fake")
    uri = artifacts.put(page_key(project.id, document.id, document.sha256, 1), b"png", "image/png")
    repo.update("documents", document.id, {"render_status": "succeeded", "page_count": 1, "page_uris": {"1": uri}})
    service.publish_task_spec(project.id, "people", SCHEMA)
    service.publish_prompt(project.id, "extract", "Return JSON", "Read {document_name} page {page_number}")
    judge_prompt = service.publish_prompt(
        project.id, "judge", "Judge carefully",
        "Prediction: {predicted} Expected: {expected} Scores: {deterministic_scores}", stage="judge",
    )
    model = service.publish_model(project.id, "fake", "fake", "fake-model")
    service.test_model(model.id)
    service.publish_tool_profile(project.id, "safe", ["validate_schema"])
    imported = service.import_ground_truth(
        project.id, f'document_id,page_number,name\n{document.id},1,Alice\n'.encode(), "truth.csv",
    )[0]
    service.revise_ground_truth(imported.id, {"name": "Alice"}, approved=True)
    run = service.create_run(
        project.id, preflight=True, llm_judge_enabled=True,
        judge_model_profile_version_id=model.id, judge_prompt_version_id=judge_prompt.id,
    )
    repo.queue_run(run.id)
    worker.process_run(run.id)
    evaluation = repo.find("evaluations", {"run_id": run.id})[0]
    assert evaluation["deterministic"]["overall_score"] == 1.0
    assert evaluation["llm_judge"]["status"] == "completed"
    assert evaluation["llm_judge"]["result"]["score"] == 1.0
