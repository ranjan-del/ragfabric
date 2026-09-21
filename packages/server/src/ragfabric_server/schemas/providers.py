"""Schemas for the provider configuration endpoint.

Note what is absent: there is no field anywhere here that carries an API key.
Secrets live in the environment (``ragfabric_core.config``), and this endpoint
only ever reports whether the environment variable a provider needs is set.
Adding a key field would turn a selection endpoint into a secret store, which
is a different thing with different obligations.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ProviderStatus(BaseModel):
    """What is selected, and whether its secret is present. Never the secret."""

    provider: str
    model: str | None
    base_url: str | None
    dim: int | None = None
    # The environment variable this provider reads its key from, or None when
    # it needs no key at all (ollama on localhost, the offline doubles).
    key_env_var: str | None
    requires_key: bool
    has_key: bool


class ProviderConfigOut(BaseModel):
    llm: ProviderStatus
    embeddings: ProviderStatus


class ProviderConfigWritten(ProviderConfigOut):
    # True when the embedding provider, model or dimension moved. ADR 0006
    # pins the dimension of an index, so this is not a hint: existing vectors
    # are unusable until the corpus is re-indexed.
    requires_reindex: bool
    # Providers are built once at application startup, so a write takes
    # effect for new processes. Saying so is better than pretending the
    # running workers picked it up.
    restart_required: bool
    changed: list[str]


class LLMProviderUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: Literal["openai", "anthropic", "ollama", "offline"]
    model: str | None = None
    base_url: str | None = None


class EmbeddingsProviderUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: Literal["openai", "ollama", "offline"]
    model: str | None = None
    dim: int | None = Field(default=None, ge=1)
    base_url: str | None = None


class ProviderConfigUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    llm: LLMProviderUpdate | None = None
    embeddings: EmbeddingsProviderUpdate | None = None

    @model_validator(mode="after")
    def _at_least_one_section(self) -> ProviderConfigUpdate:
        if self.llm is None and self.embeddings is None:
            raise ValueError("Name at least one of 'llm' or 'embeddings'.")
        return self


class ProviderTestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target: Literal["llm", "embeddings"]


class ProviderTestResult(BaseModel):
    target: str
    ok: bool
    # Empty on success. On failure, the provider's own error text, which is
    # reported as it is rather than flattened into "connection failed".
    detail: str
    model: str | None
