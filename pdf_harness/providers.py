"""Administrator-owned LLM connector registry."""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ConnectorDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    label: str
    base_url: str
    secret_ref: str
    allowed_models: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def secure_endpoint(self) -> "ConnectorDefinition":
        parsed = urlparse(self.base_url)
        if parsed.scheme != "https" or not parsed.hostname:
            raise ValueError("connector base_url must be HTTPS")
        if not (self.secret_ref.startswith("projects/") or self.secret_ref == "env:STANFORD_API_KEY"):
            raise ValueError("connector secret_ref must be a Secret Manager resource (or local Stanford key)")
        return self


def load_connectors(raw_json: str | None, production: bool = False) -> dict[str, ConnectorDefinition]:
    if raw_json:
        raw: list[dict[str, Any]] = json.loads(raw_json)
    elif not production:
        raw = [{
            "id": "stanford", "label": "Stanford AI API",
            "base_url": "https://aiapi-prod.stanford.edu/v1",
            "secret_ref": "env:STANFORD_API_KEY",
            "allowed_models": ["gpt-5-mini", "claude-sonnet-4-6", "gemini-2.5-flash"],
        }]
    else:
        raise ValueError("HARNESS_LLM_CONNECTORS_JSON is required in production")
    connectors = {item.id: item for item in (ConnectorDefinition.model_validate(value) for value in raw)}
    if len(connectors) != len(raw):
        raise ValueError("connector IDs must be unique")
    if production and any(not item.secret_ref.startswith("projects/") for item in connectors.values()):
        raise ValueError("production connectors must use Secret Manager resource references")
    return connectors
