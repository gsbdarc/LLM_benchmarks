from pdf_harness.dispatch import CloudRunJobDispatcher


def test_cloud_run_dispatch_builds_override_request(monkeypatch):
    from google.cloud import run_v2

    captured = {}

    class Operation:
        class Inner:
            name = "operations/123"
        operation = Inner()

    class Client:
        def run_job(self, request):
            captured["request"] = request
            return Operation()

    monkeypatch.setattr(run_v2, "JobsClient", Client)
    dispatcher = CloudRunJobDispatcher("project", "us-west1", "worker")
    assert dispatcher.dispatch("run-id") == "operations/123"
    request = captured["request"]
    assert request.name == "projects/project/locations/us-west1/jobs/worker"
    assert request.overrides.container_overrides[0].env[0].name == "HARNESS_RUN_ID"
    assert request.overrides.container_overrides[0].env[0].value == "run-id"
