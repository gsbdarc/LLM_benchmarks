"""Environment-backed application configuration with no import-time network calls."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


class Settings(BaseModel):
    repository_backend: Literal["mongo", "memory"] = "mongo"
    mongo_uri: str = "mongodb://localhost:27017"
    mongo_db: str = "pdf_extraction_harness"
    artifact_backend: str = "local"
    local_data_dir: Path = ROOT / ".harness-data"
    gcs_bucket: str | None = None
    gcp_project: str | None = None
    gcp_region: str = "us-west1"
    cloud_run_job: str = "pdf-harness-worker"
    dispatch_url: str = "http://localhost:8080"
    app_password: str | None = None
    app_password_secret: str | None = None
    worker_concurrency: int = Field(default=4, ge=1, le=64)
    worker_max_attempts: int = Field(default=3, ge=1, le=10)
    worker_lease_seconds: int = Field(default=900, ge=60)
    max_upload_mb: int = Field(default=200, ge=1, le=5000)
    dispatcher_audience: str | None = None
    weave_project: str | None = None
    llm_connectors_json: str | None = None
    environment: str = "development"

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            repository_backend=os.getenv("HARNESS_REPOSITORY_BACKEND", "mongo"),
            mongo_uri=os.getenv("HARNESS_MONGO_URI", "mongodb://localhost:27017"),
            mongo_db=os.getenv("HARNESS_MONGO_DB", "pdf_extraction_harness"),
            artifact_backend=os.getenv("HARNESS_ARTIFACT_BACKEND", "local"),
            local_data_dir=Path(os.getenv("HARNESS_LOCAL_DATA_DIR", str(ROOT / ".harness-data"))),
            gcs_bucket=os.getenv("HARNESS_GCS_BUCKET"),
            gcp_project=os.getenv("HARNESS_GCP_PROJECT"),
            gcp_region=os.getenv("HARNESS_GCP_REGION", "us-west1"),
            cloud_run_job=os.getenv("HARNESS_CLOUD_RUN_JOB", "pdf-harness-worker"),
            dispatch_url=os.getenv("HARNESS_DISPATCH_URL", "http://localhost:8080"),
            app_password=os.getenv("HARNESS_APP_PASSWORD"),
            app_password_secret=os.getenv("HARNESS_APP_PASSWORD_SECRET"),
            worker_concurrency=int(os.getenv("HARNESS_WORKER_CONCURRENCY", "4")),
            worker_max_attempts=int(os.getenv("HARNESS_WORKER_MAX_ATTEMPTS", "3")),
            worker_lease_seconds=int(os.getenv("HARNESS_WORKER_LEASE_SECONDS", "900")),
            max_upload_mb=int(os.getenv("HARNESS_MAX_UPLOAD_MB", "200")),
            dispatcher_audience=os.getenv("HARNESS_DISPATCH_AUDIENCE"),
            weave_project=os.getenv("HARNESS_WEAVE_PROJECT"),
            llm_connectors_json=os.getenv("HARNESS_LLM_CONNECTORS_JSON"),
            environment=os.getenv("HARNESS_ENVIRONMENT", "development"),
        )


settings = Settings.from_env()
