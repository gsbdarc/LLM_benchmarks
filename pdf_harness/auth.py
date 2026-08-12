"""Prototype password authentication behind a replaceable provider boundary."""

from __future__ import annotations

import hmac
from typing import Protocol


class AuthProvider(Protocol):
    def authenticate(self, credential: str) -> bool: ...


class SharedPasswordAuth:
    def __init__(self, expected_password: str) -> None:
        if not expected_password:
            raise ValueError("shared app password is not configured")
        self.expected = expected_password.encode()

    def authenticate(self, credential: str) -> bool:
        return hmac.compare_digest(self.expected, (credential or "").encode())
