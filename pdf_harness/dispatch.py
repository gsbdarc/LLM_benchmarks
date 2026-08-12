"""Cloud Run Job dispatcher and local development dispatcher."""

from __future__ import annotations

from typing import Protocol


class Dispatcher(Protocol):
    def health(self) -> tuple[bool, str]: ...
    def dispatch(self, run_id: str) -> str: ...


class LocalDispatcher:
    def __init__(self, worker) -> None:
        self.worker = worker

    def health(self) -> tuple[bool, str]:
        return True, "local synchronous worker available"

    def dispatch(self, run_id: str) -> str:
        self.worker.process_run(run_id)
        return f"local:{run_id}"


class CloudRunJobDispatcher:
    def __init__(self, project: str, region: str, job: str) -> None:
        self.project = project
        self.region = region
        self.job = job

    @property
    def name(self) -> str:
        return f"projects/{self.project}/locations/{self.region}/jobs/{self.job}"

    def health(self) -> tuple[bool, str]:
        try:
            from google.cloud import run_v2

            run_v2.JobsClient().get_job(name=self.name)
            return True, f"Cloud Run Job {self.job} reachable"
        except Exception as exc:  # noqa: BLE001
            return False, f"Cloud Run Job unavailable: {type(exc).__name__}"

    def dispatch(self, run_id: str) -> str:
        from google.cloud import run_v2

        overrides = run_v2.RunJobRequest.Overrides(
            container_overrides=[run_v2.RunJobRequest.Overrides.ContainerOverride(
                env=[run_v2.EnvVar(name="HARNESS_RUN_ID", value=run_id)]
            )]
        )
        operation = run_v2.JobsClient().run_job(
            request=run_v2.RunJobRequest(name=self.name, overrides=overrides)
        )
        return operation.operation.name
