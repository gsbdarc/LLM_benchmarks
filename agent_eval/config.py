"""
config.py — backend selection, OpenAI client construction, and shared paths.

Two behaviours worth calling out:
  * `reasoning_level()` reads the configured reasoning effort out of a model's
    completion kwargs so it can be logged as a flat field (plan §2).
  * Backends live in `eval/backends/<endpoint>.json` (data, not code) — ONE file
    per endpoint (the filename is the endpoint name). Each file declares the
    endpoint once and lists its models in an int-keyed `models` map. Any endpoint
    can therefore run many models with no code change; a run selects one via
    `--model <int-key>` (or a raw model-id override). This separates the ENDPOINT
    (URL/framework/auth — few, stable) from the MODELS (many) so the config scales
    cleanly.

Per-endpoint file schema (eval/backends/<endpoint>.json):
    framework         serving engine label: "openai" | "nim" | "vllm" | "ollama"
    base_url          OpenAI-compatible endpoint (…/v1)
    api_key_env       env var holding the key (optional; omit for keyless local servers)
    models            { "<int>": { "model": <id>, "completion_kwargs": {...} }, ... }

The int model key is the AGENT/eval model (the judge) — distinct from `model_id`
in the run rows, which is the model being evaluated. The default model for an
endpoint is the lowest int key.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI, OpenAI

# agent_eval/ (package dir) -> repo root
PKG_DIR = Path(__file__).resolve().parent
REPO_ROOT = PKG_DIR.parent

WEAVE_PROJECT = "darc/metric-eval-agent"
MAX_STEPS = 12

# Where the flat run-summary Parquet dataset lives (the OLAP layer).
AGENT_RUNS_DIR = REPO_ROOT / "outputs" / "agent_runs"

# Data-driven backend registry: one JSON file per endpoint (see module docstring).
BACKENDS_DIR = Path(__file__).resolve().parent / "backends"


def _load_backends(directory: Path = BACKENDS_DIR) -> dict:
    """Load every eval/backends/<endpoint>.json into {endpoint_name: config}."""
    if not directory.is_dir():
        raise FileNotFoundError(
            f"backend registry dir not found at {directory} — expected eval/backends/"
        )
    registry: dict = {}
    for path in sorted(directory.glob("*.json")):
        try:
            registry[path.stem] = json.loads(path.read_text())
        except json.JSONDecodeError as e:
            raise ValueError(f"backend file {path} is not valid JSON: {e}") from e
    if not registry:
        raise FileNotFoundError(f"no backend JSON files found in {directory}")
    return registry


BACKENDS = _load_backends()


def _default_model_key(cfg: dict) -> str:
    """The default model key for an endpoint: the lowest integer key."""
    models = cfg.get("models") or {}
    if not models:
        raise ValueError("backend has no models defined")
    return min(models, key=lambda k: int(k))


def resolve_model(backend: str, model: str | int | None = None) -> tuple[str, dict, int | None]:
    """Resolve a (model_string, completion_kwargs, model_key) for a backend.

    `model` may be an int model KEY from the endpoint's `models` map, None for the
    endpoint default (lowest key), or a raw model-id override (used verbatim with
    the default model's completion_kwargs; model_key is then None).
    """
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; choose from {list(BACKENDS)}")
    models = BACKENDS[backend].get("models") or {}
    key = _default_model_key(BACKENDS[backend]) if model is None else str(model)
    if key in models:
        entry = models[key]
        return entry["model"], entry.get("completion_kwargs", {}), int(key)
    # A model *string* (not a key) that matches a configured model resolves to that
    # key — so the mapping storing "gpt-5-mini" still yields agent_model_key=1.
    for k, entry in models.items():
        if entry.get("model") == str(model):
            return entry["model"], entry.get("completion_kwargs", {}), int(k)
    # Genuine raw model-id override: keep the default model's kwargs, no numeric key.
    default = models[_default_model_key(BACKENDS[backend])]
    return str(model), default.get("completion_kwargs", {}), None


def model_price(backend: str, model: str | int | None = None) -> tuple[float | None, float | None]:
    """(input_price, output_price) in USD per 1M tokens for a backend model.

    Mirrors resolve_model's key/string resolution. Returns (None, None) for an
    unknown backend, an unpriced model, or a raw model-id override — so cost is
    simply left null rather than guessed. Local models are priced 0/0.
    """
    if backend not in BACKENDS:
        return None, None
    models = BACKENDS[backend].get("models") or {}
    key = _default_model_key(BACKENDS[backend]) if model is None else str(model)
    entry = models.get(key)
    if entry is None:
        entry = next((e for e in models.values() if e.get("model") == str(model)), None)
    if entry is None:
        return None, None
    return entry.get("input_price"), entry.get("output_price")


def _read_api_key(cfg: dict) -> str:
    """Resolve an API key from the configured file, then env var, else 'not-used'."""
    key_file = cfg.get("api_key_file")
    if key_file:
        # Try relative to mcp/ first (where the notebook ran), then repo root.
        for base in (PKG_DIR, REPO_ROOT):
            p = base / key_file
            if p.exists():
                return p.read_text().strip()
    env_name = cfg.get("api_key_env")
    if env_name and os.getenv(env_name):
        return os.environ[env_name]
    return "not-used"


def reasoning_level(backend: str, completion_kwargs: dict | None = None) -> str:
    """A flat, comparable label for how much reasoning the model was asked to do.

    - playground: the configured `reasoning_effort` (e.g. "high").
    - nim/Gemma: "thinking" / "no-thinking" from the enable_thinking template flag.

    Falls back to the endpoint's default model's kwargs when none are passed.
    """
    if completion_kwargs is not None:
        kw = completion_kwargs
    elif backend in BACKENDS:
        _, kw, _ = resolve_model(backend)
    else:
        kw = {}
    if "reasoning_effort" in kw:
        return str(kw["reasoning_effort"])
    enable = (
        kw.get("extra_body", {})
        .get("chat_template_kwargs", {})
        .get("enable_thinking")
    )
    if enable is True:
        return "thinking"
    if enable is False:
        return "no-thinking"
    return "default"


def metrics_url(base_url: str) -> str:
    """Map an OpenAI base_url (…/v1) to the vLLM/NIM Prometheus endpoint (…/metrics).

    NIM exposes Prometheus text at /v1/metrics (confirmed in nim-server-*.out).
    """
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        return base + "/metrics"
    return base + "/v1/metrics"


def build_backend(backend: str, model: str | int | None = None) -> tuple[Any, str, dict, str]:
    """Return (client, model, completion_kwargs, base_url) for a backend name.

    `model` selects a model on the endpoint: an int model KEY, the endpoint
    default (None), or a raw model-id override. One endpoint can thus serve many
    models with no new file.

    The client is **AsyncOpenAI**: the agent loop awaits it so that N concurrent
    runs actually parallelize at the LLM layer (a blocking sync call never yields
    the event loop, which is what capped local-model throughput at ~1 in-flight
    request). Sync one-shot scripts should use `sync_openai_client` instead.
    """
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; choose from {list(BACKENDS)}")
    cfg = BACKENDS[backend]
    resolved_model, completion_kwargs, _ = resolve_model(backend, model)
    client = AsyncOpenAI(base_url=cfg["base_url"], api_key=_read_api_key(cfg))
    return client, resolved_model, completion_kwargs, cfg["base_url"]


def sync_openai_client(backend: str, model: str | int | None = None, timeout: float | None = None) -> tuple[Any, str, dict]:
    """A synchronous OpenAI client for a backend — for one-shot scripts.

    Used by the dashboard builder (a sync, run-once tool) to write path summaries.
    `build_backend` is the agent's client (async in Part 6, for concurrent runs);
    this stays sync so the build script needs no event loop. `timeout` (seconds)
    makes calls fail fast when the endpoint is down so callers can degrade gracefully.
    Returns (client, model, completion_kwargs).
    """
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; choose from {list(BACKENDS)}")
    cfg = BACKENDS[backend]
    resolved_model, completion_kwargs, _ = resolve_model(backend, model)
    kwargs = {"timeout": timeout} if timeout is not None else {}
    client = OpenAI(base_url=cfg["base_url"], api_key=_read_api_key(cfg), **kwargs)
    return client, resolved_model, completion_kwargs


def framework(backend: str) -> str | None:
    """The serving-engine label for a backend (openai/nim/vllm/ollama)."""
    return BACKENDS.get(backend, {}).get("framework")


def gpu_type(explicit: str | None = None) -> str | None:
    """GPU model of the server hosting the model under test.

    The eval client and the GPU server are different hosts, so we never read
    nvidia-smi here — the value is propagated explicitly: an explicit arg (the
    CLI --gpu-type) wins, else the GPU_TYPE env var (echoed by
    run_nim_server.slurm), else None (e.g. hosted/playground backends).
    """
    if explicit:
        return explicit.strip()
    val = os.getenv("GPU_TYPE")
    return val.strip() if val and val.strip() else None
