"""AuthProvider: turns a credential (JWT, API key, OIDC token) into a Principal."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ragfabric_core.auth.principal import Principal


@runtime_checkable
class AuthProvider(Protocol):
    name: str

    def authenticate(self, credential: str) -> Principal | None: ...
