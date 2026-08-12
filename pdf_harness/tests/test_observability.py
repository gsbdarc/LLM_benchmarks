from pdf_harness.observability import NullObserver, build_observer
from pdf_harness.worker import _safe_payload


def test_observability_is_disabled_without_key(monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    observer = build_observer("team/project")
    assert isinstance(observer, NullObserver)
    assert observer.record("run", {"ok": True}) is None


def test_trace_payload_redacts_secrets_and_bounds_content():
    result = _safe_payload({
        "api_key": "secret", "access_token": "secret", "value": "x" * 5000,
        "nested": {"password": "secret"},
    })
    assert result["api_key"] == "[REDACTED]"
    assert result["access_token"] == "[REDACTED]"
    assert result["nested"]["password"] == "[REDACTED]"
    assert result["value"].endswith("...[truncated]")
