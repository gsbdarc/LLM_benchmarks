"""Best-effort observability mirror; never part of the durable workflow state."""

from __future__ import annotations

import os
from typing import Any, Protocol


class Observer(Protocol):
    def record(self, event: str, attributes: dict[str, Any]) -> str | None: ...


class NullObserver:
    def record(self, event: str, attributes: dict[str, Any]) -> str | None:
        return None


class WeaveObserver:
    """Send sanitized metadata to W&B Weave without coupling run success to tracing."""

    def __init__(self, project: str) -> None:
        import weave

        self.weave = weave
        self.client = weave.init(project)

        @weave.op(name="pdf_harness.event", enable_code_capture=False)
        def emit(event_name: str, metadata: dict[str, Any]) -> dict[str, Any]:
            return {"recorded": True, "event": event_name, **metadata}

        self.emit = emit

    def record(self, event: str, attributes: dict[str, Any]) -> str | None:
        try:
            self.emit(event, attributes)
            call = self.weave.get_current_call()
            return getattr(call, "ui_url", None) if call else None
        except Exception:  # noqa: BLE001 — observability cannot fail extraction
            return None


def build_observer(project: str | None) -> Observer:
    if not project or not os.getenv("WANDB_API_KEY") or os.getenv("EVAL_DISABLE_WEAVE") == "1":
        return NullObserver()
    try:
        return WeaveObserver(project)
    except Exception:  # noqa: BLE001
        return NullObserver()
