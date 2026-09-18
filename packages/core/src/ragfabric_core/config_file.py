"""ragfabric.yaml: everything that is not a secret.

Secrets stay in the environment (see config.py Settings). This file chooses
implementations and tunes them. Unknown keys are errors, because a typo that
silently falls back to a default is the worst kind of configuration bug.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

ENV_VAR = "RAGFABRIC_CONFIG"
DEFAULT_FILENAME = "ragfabric.yaml"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LLMConfig(_Strict):
    provider: Literal["openai", "anthropic", "ollama", "offline"] = "ollama"
    model: str | None = "llama3.2:3b"
    base_url: str | None = "http://localhost:11434/v1"


class EmbeddingsConfig(_Strict):
    provider: Literal["openai", "ollama", "offline"] = "ollama"
    model: str | None = "nomic-embed-text"
    dim: int | None = Field(default=768, ge=1)
    base_url: str | None = "http://localhost:11434/v1"


class RerankerConfig(_Strict):
    kind: Literal["none", "llm", "cross_encoder"] = "none"


class VectorStoreConfig(_Strict):
    kind: Literal["pgvector", "chroma", "memory"] = "pgvector"


class LexicalStoreConfig(_Strict):
    kind: Literal["postgres_fts", "bm25"] = "postgres_fts"


class GraphStoreConfig(_Strict):
    kind: Literal["neo4j", "none"] = "neo4j"
    enabled: bool = False


class CacheConfig(_Strict):
    kind: Literal["redis", "memory"] = "redis"


class IngestionConfig(_Strict):
    chunk_size: int = Field(default=600, ge=50)
    chunk_overlap: int = Field(default=80, ge=0)
    retain_originals: bool = True
    uploads_dir: str = "data/uploads"
    # inline: index in the request that ingests. queue: enqueue for `ragfabric worker`.
    indexing: Literal["inline", "queue"] = "inline"


class StrategiesConfig(_Strict):
    traditional: dict[str, float | int | str | bool] = Field(
        default_factory=lambda: {
            "top_k": 8,
            "similarity_threshold": 0.25,
            "rerank": "none",
            "max_context_tokens": 6000,
        }
    )
    vectorless: dict[str, float | int | str | bool] = Field(
        default_factory=lambda: {"top_k": 8, "phrase_boost": 2.0, "identifier_boost": 3.0}
    )
    agentic: dict[str, float | int | str | bool] = Field(
        default_factory=lambda: {"max_iterations": 4, "max_cost_usd": 0.10, "max_latency_ms": 30000}
    )
    graph: dict[str, float | int | str | bool] = Field(
        default_factory=lambda: {"max_hops": 2, "max_nodes": 200}
    )


class RouterConfig(_Strict):
    mode: Literal["auto", "manual"] = "auto"
    min_confidence: float = Field(default=0.6, ge=0.0, le=1.0)
    classifier_model: str | None = None


class LimitsConfig(_Strict):
    max_upload_mb: int = Field(default=50, ge=1)
    allowed_types: list[str] = Field(
        default_factory=lambda: ["pdf", "docx", "pptx", "txt", "csv", "md"]
    )
    rate_limit_per_minute: int = Field(default=60, ge=1)


class TelemetryConfig(_Strict):
    otlp_endpoint: str | None = None


class RagFabricConfig(_Strict):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    lexical_store: LexicalStoreConfig = Field(default_factory=LexicalStoreConfig)
    graph_store: GraphStoreConfig = Field(default_factory=GraphStoreConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    strategies: StrategiesConfig = Field(default_factory=StrategiesConfig)
    router: RouterConfig = Field(default_factory=RouterConfig)
    limits: LimitsConfig = Field(default_factory=LimitsConfig)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)


def resolve_config_path(path: Path | None = None) -> Path | None:
    if path is not None:
        return Path(path)
    from_env = os.environ.get(ENV_VAR)
    if from_env:
        return Path(from_env)
    local = Path.cwd() / DEFAULT_FILENAME
    return local if local.exists() else None


def load_config(path: Path | None = None) -> RagFabricConfig:
    resolved = resolve_config_path(path)
    if resolved is None:
        return RagFabricConfig()
    data = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    return RagFabricConfig.model_validate(data)
