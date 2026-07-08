"""Tests for eval.config — backend resolution, reasoning labels, URL mapping."""

import pytest

from agent_eval import config


def test_reasoning_level_playground():
    assert config.reasoning_level("playground") == "high"


def test_reasoning_level_nim_thinking():
    # NIM default has enable_thinking=True -> "thinking"
    assert config.reasoning_level("nim") == "thinking"


def test_reasoning_level_nim_no_thinking():
    kw = {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
    assert config.reasoning_level("nim", kw) == "no-thinking"


def test_reasoning_level_default_when_unspecified():
    assert config.reasoning_level("nim", {}) == "default"


def test_reasoning_level_explicit_kwargs_override_backend():
    assert config.reasoning_level("playground", {"reasoning_effort": "low"}) == "low"


@pytest.mark.parametrize(
    "base,expected",
    [
        ("http://yen-gpu4:8000/v1", "http://yen-gpu4:8000/v1/metrics"),
        ("http://yen-gpu4:8000/v1/", "http://yen-gpu4:8000/v1/metrics"),
        ("http://host:8000", "http://host:8000/v1/metrics"),
    ],
)
def test_metrics_url(base, expected):
    assert config.metrics_url(base) == expected


def test_read_api_key_falls_back_to_not_used(monkeypatch):
    monkeypatch.delenv("PLAYGROUND_API_KEY", raising=False)
    cfg = {"api_key_file": "does-not-exist.txt", "api_key_env": "PLAYGROUND_API_KEY"}
    assert config._read_api_key(cfg) == "not-used"


def test_read_api_key_from_env(monkeypatch):
    monkeypatch.setenv("PLAYGROUND_API_KEY", "sk-test-123")
    cfg = {"api_key_file": None, "api_key_env": "PLAYGROUND_API_KEY"}
    assert config._read_api_key(cfg) == "sk-test-123"


def test_build_backend_rejects_unknown():
    with pytest.raises(ValueError):
        config.build_backend("nonexistent")


def test_build_backend_nim_returns_model(monkeypatch):
    client, model, kwargs, base_url = config.build_backend("nim")
    assert model == "google/gemma-4-31b-it"
    assert base_url.endswith("/v1")
    assert "temperature" in kwargs


def test_build_backend_raw_model_override():
    # Raw model-id override on an endpoint — no new registry entry needed.
    _, model, _, base_url = config.build_backend("playground", model="gpt-4o")
    assert model == "gpt-4o"
    assert base_url.endswith("/v1")


def test_backends_loaded_from_dir_with_framework():
    assert set(config.BACKENDS) >= {"playground", "nim", "summarizer"}
    assert config.framework("playground") == "openai"
    assert config.framework("nim") == "nim"
    assert config.framework("nope") is None


def test_resolve_model_default_and_by_key():
    # Default (lowest int key) resolves model #1; explicit key returns its id + key.
    model, kwargs, key = config.resolve_model("nim")
    assert model == "google/gemma-4-31b-it"
    assert key == 1
    assert "temperature" in kwargs
    model2, _, key2 = config.resolve_model("nim", 1)
    assert (model2, key2) == ("google/gemma-4-31b-it", 1)


def test_resolve_model_raw_override_has_no_key():
    model, _, key = config.resolve_model("playground", "gpt-4o")
    assert model == "gpt-4o"
    assert key is None


def test_resolve_model_string_matches_configured_key():
    # Passing the model STRING (as the eval mapping stores it) still yields the key.
    model, _, key = config.resolve_model("playground", "gpt-5-mini")
    assert model == "gpt-5-mini"
    assert key == 1


def test_gpu_type_explicit_wins(monkeypatch):
    monkeypatch.setenv("GPU_TYPE", "NVIDIA H200")
    assert config.gpu_type("NVIDIA A100") == "NVIDIA A100"


def test_gpu_type_from_env(monkeypatch):
    monkeypatch.setenv("GPU_TYPE", "  NVIDIA H200  ")
    assert config.gpu_type() == "NVIDIA H200"


def test_gpu_type_none_when_unset(monkeypatch):
    monkeypatch.delenv("GPU_TYPE", raising=False)
    assert config.gpu_type() is None
