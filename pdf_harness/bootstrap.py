"""Dependency construction shared by Streamlit, the function, and Cloud Run Job."""

from __future__ import annotations

from dataclasses import dataclass

from .config import Settings, settings
from .dispatch import CloudRunJobDispatcher, Dispatcher, LocalDispatcher
from .llm import OpenAICompatibleExtractionClient
from .observability import build_observer
from .providers import load_connectors
from .repository import InMemoryRepository, MongoRepository, Repository
from .secrets import CompositeSecretResolver, SecretResolver
from .service import HarnessService
from .storage import ArtifactStore, GCSArtifactStore, LocalArtifactStore
from .worker import HarnessWorker


@dataclass
class AppContext:
    settings: Settings
    repo: Repository
    artifacts: ArtifactStore
    secrets: SecretResolver
    worker: HarnessWorker
    dispatcher: Dispatcher
    service: HarnessService


def build_context(config: Settings | None = None) -> AppContext:
    cfg = config or settings
    if cfg.repository_backend == "memory":
        if cfg.environment == "production":
            raise ValueError("the in-memory repository is only available in development")
        repo: Repository = InMemoryRepository()
    else:
        repo = MongoRepository(cfg.mongo_uri, cfg.mongo_db)
    artifacts: ArtifactStore
    if cfg.artifact_backend == "gcs":
        if not cfg.gcs_bucket:
            raise ValueError("HARNESS_GCS_BUCKET is required for the gcs artifact backend")
        artifacts = GCSArtifactStore(cfg.gcs_bucket)
    else:
        artifacts = LocalArtifactStore(cfg.local_data_dir)
    secrets = CompositeSecretResolver()
    llm = OpenAICompatibleExtractionClient()
    connectors = load_connectors(cfg.llm_connectors_json, production=cfg.environment == "production")
    worker = HarnessWorker(
        repo, artifacts, secrets, llm, cfg.worker_concurrency,
        cfg.worker_max_attempts, cfg.worker_lease_seconds, build_observer(cfg.weave_project),
    )
    if cfg.environment == "production":
        if not cfg.gcp_project:
            raise ValueError("HARNESS_GCP_PROJECT is required in production")
        dispatcher: Dispatcher = CloudRunJobDispatcher(cfg.gcp_project, cfg.gcp_region, cfg.cloud_run_job)
    else:
        dispatcher = LocalDispatcher(worker)
    service = HarnessService(
        repo, artifacts, secrets, llm, dispatcher.health,
        max_upload_bytes=cfg.max_upload_mb * 1024 * 1024,
        connectors=connectors,
    )
    return AppContext(cfg, repo, artifacts, secrets, worker, dispatcher, service)
