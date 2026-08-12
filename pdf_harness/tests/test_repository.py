from datetime import timedelta

from pdf_harness.models import Extraction, Project, Run, RunSnapshot, RunStatus, Stage, WorkItem, WorkStatus, utcnow
from pdf_harness.repository import InMemoryRepository


def snapshot():
    return RunSnapshot(
        project_id="p", mode="extraction", task_spec_version_id="s",
        prompt_version_ids=["prompt"], model_profile_version_ids=["model"], document_ids=["doc"],
    )


def test_queue_is_atomic_and_status_scoped():
    repo = InMemoryRepository()
    run = Run(project_id="p", snapshot=snapshot(), status=RunStatus.PREFLIGHT_READY)
    repo.put("runs", run)
    assert repo.queue_run(run.id)
    assert not repo.queue_run(run.id)
    assert repo.get("runs", run.id)["status"] == RunStatus.DISPATCHING.value


def test_claim_reclaims_expired_lease():
    repo = InMemoryRepository()
    work = WorkItem(run_id="r", project_id="p", stage=Stage.EXTRACT, document_id="d")
    repo.put("work_items", work)
    first = repo.claim_work("r", "worker-1", 60)
    assert first["attempt"] == 1
    assert repo.claim_work("r", "worker-2", 60) is None
    repo.update("work_items", work.id, {"lease_expires_at": utcnow() - timedelta(seconds=1)})
    second = repo.claim_work("r", "worker-2", 60)
    assert second["attempt"] == 2
    assert second["lease_owner"] == "worker-2"


def test_idempotent_entities_keep_original_identity():
    repo = InMemoryRepository()
    work = WorkItem(run_id="r", project_id="p", stage=Stage.EXTRACT, document_id="d")
    first = repo.put("work_items", work)
    duplicate = WorkItem(run_id="r", project_id="p", stage=Stage.EXTRACT, document_id="d")
    second = repo.put("work_items", duplicate)
    assert second["id"] == first["id"]
    extraction = Extraction(
        work_item_id=first["id"], run_id="r", project_id="p", document_id="d",
        model_profile_version_id="m", prompt_version_id="q",
    )
    saved = repo.put("extractions", extraction)
    replacement = extraction.model_copy(update={"id": "new-id", "schema_valid": True})
    resaved = repo.put("extractions", replacement)
    assert resaved["id"] == saved["id"]
    assert len(repo.find("extractions")) == 1
