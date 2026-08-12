"""Secret resolution. Mongo stores references, never secret values."""

from __future__ import annotations

import os
from typing import Protocol


class SecretResolver(Protocol):
    def get(self, reference: str) -> str: ...


class EnvironmentSecretResolver:
    """Resolve `env:NAME` or a bare environment variable name."""

    def get(self, reference: str) -> str:
        name = reference.removeprefix("env:")
        value = os.getenv(name)
        if not value:
            raise KeyError(f"secret environment variable {name!r} is not configured")
        return value


class GoogleSecretResolver:
    def __init__(self) -> None:
        from google.cloud import secretmanager

        self.client = secretmanager.SecretManagerServiceClient()

    def get(self, reference: str) -> str:
        response = self.client.access_secret_version(request={"name": reference})
        return response.payload.data.decode("utf-8")


class CompositeSecretResolver:
    def __init__(self) -> None:
        self._env = EnvironmentSecretResolver()
        self._google: GoogleSecretResolver | None = None

    def get(self, reference: str) -> str:
        if reference.startswith("projects/"):
            self._google = self._google or GoogleSecretResolver()
            return self._google.get(reference)
        return self._env.get(reference)
