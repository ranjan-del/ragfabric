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
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ragfabric_core.agent.state import NodeName
from ragfabric_core.graph.contracts import EntityType, RelationType
from ragfabric_core.graph.traverse import DEFAULT_NODE_BUDGET, MAX_HOPS_CEILING

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
    kind: Literal["postgres_fts", "bm25", "bm25_memory"] = "postgres_fts"
    # Only meaningful for bm25_memory, which holds the corpus in each worker's
    # own memory. Exceeding it raises rather than silently answering from part
    # of the corpus. See stores/bm25_memory.py for why this is not optional.
    max_chunks: int = Field(default=50_000, ge=1)


def _each_once(values: list) -> list:
    if len(set(values)) != len(values):
        raise ValueError("each type may be listed once")
    return values


class GraphStoreConfig(_Strict):
    """Graph extraction at ingest time. Every field is read by the extract_graph handler.

    ``enabled`` defaults to false so a deployment that does not want a graph
    pays nothing: the handler returns before building a prompt, and no model
    provider is constructed for it.

    ``kind`` has one value. The graph lives in the relational database and is
    walked with recursive CTEs (ADR 0011); the Neo4j backend promised in v0.2
    was never built, and a config that still names it is refused rather than
    silently served by something else.

    ``confidence_floor`` and ``similarity_threshold`` are untuned: 0.5 and 0.9
    are starting points, not measured optima, until the Phase 8 evaluation
    measures extraction precision and merge precision against them.

    ``entity_types`` and ``relation_types`` narrow extraction: a disabled type
    is removed from the prompt, and anything the model still reports under it
    is discarded and counted, never stored.
    """

    enabled: bool = False
    kind: Literal["postgres"] = "postgres"
    extraction_model: str | None = None
    confidence_floor: float = Field(default=0.5, ge=0.0, le=1.0)
    similarity_threshold: float = Field(default=0.9, ge=0.0, le=1.0)
    entity_types: list[EntityType] = Field(default_factory=lambda: list(EntityType), min_length=1)
    relation_types: list[RelationType] = Field(
        default_factory=lambda: list(RelationType), min_length=1
    )

    @field_validator("kind", mode="before")
    @classmethod
    def _neo4j_is_gone(cls, kind: object) -> object:
        if kind == "neo4j":
            raise ValueError(
                "graph_store.kind neo4j is no longer supported: the graph lives in PostgreSQL "
                "(ADR 0011). Use kind: postgres, or omit the key"
            )
        return kind

    @field_validator("entity_types", "relation_types")
    @classmethod
    def _no_duplicate_types(cls, values: list) -> list:
        return _each_once(values)


class CacheConfig(_Strict):
    kind: Literal["redis", "memory"] = "redis"


class IngestionConfig(_Strict):
    chunk_size: int = Field(default=600, ge=50)
    chunk_overlap: int = Field(default=80, ge=0)
    retain_originals: bool = True
    uploads_dir: str = "data/uploads"
    # inline: index in the request that ingests. queue: enqueue for `ragfabric worker`.
    indexing: Literal["inline", "queue"] = "inline"


class VectorlessConfig(_Strict):
    """Tuning for the vectorless strategy. Every bound below is deliberate.

    ``b`` is constrained to [0, 1] because outside that range the length
    normalisation term stops meaning anything: a negative b rewards long
    chunks, and b > 1 can drive the BM25 denominator negative, which flips the
    sign of the score. ``phrase_boost`` and ``identifier_boost`` have a floor
    of 1.0 because a multiplier below one is a penalty, and anyone who wants
    to penalise exact matches should have to say so with a different key
    rather than by writing a number that looks like a boost.
    """

    top_k: int = Field(default=8, ge=1)
    k1: float = Field(default=1.2, ge=0.0)
    b: float = Field(default=0.75, ge=0.0, le=1.0)
    phrase_boost: float = Field(default=2.0, ge=1.0)
    identifier_boost: float = Field(default=3.0, ge=1.0)
    fusion_k: int = Field(default=60, ge=1)
    fusion_weights: tuple[float, float] = (1.0, 1.0)
    max_context_tokens: int = Field(default=6000, ge=100)


AgentToolName = Literal["semantic_search", "lexical_search", "fetch_document"]

DEFAULT_TOOLS: list[AgentToolName] = ["semantic_search", "lexical_search", "fetch_document"]


class PerNodeLLMCallsConfig(_Strict):
    """How many model calls each node may make across the whole run.

    Only the four nodes that call a model appear. ``retrieve`` and ``finalize``
    make none, and a cap on a node that cannot spend would be a number in a file
    that never does anything, which is the kind of setting people later tune in
    the belief that it matters.

    Why per-node caps exist at all: a global cap alone lets one runaway node
    consume the entire budget before the others ever run, so a plan that loops
    would leave nothing for assess and the answer would be generated from
    evidence nobody judged.
    """

    plan: int = Field(default=2, ge=1)
    assess: int = Field(default=6, ge=1)
    repair: int = Field(default=6, ge=1)
    generate: int = Field(default=2, ge=1)


class AgenticConfig(_Strict):
    """Everything that bounds the agent. Typed, floored, and strict about typos.

    This is the one strategy that decides for itself how much work to do, so
    each limit is a declared field with a floor rather than a key in a loose
    dict. A misspelled cap in a dict does not raise: it leaves the default in
    place and the operator finds out from the bill.
    """

    max_iterations: int = Field(default=4, ge=1)
    max_llm_calls: int = Field(default=12, ge=1)
    per_node_llm_calls: PerNodeLLMCallsConfig = Field(default_factory=PerNodeLLMCallsConfig)
    max_cost_usd: float = Field(default=0.10, ge=0.0)
    max_latency_ms: int = Field(default=30_000, ge=1)
    tools: list[AgentToolName] = Field(default_factory=lambda: list(DEFAULT_TOOLS), min_length=1)
    assess_strictness: Literal["strict", "lenient"] = "strict"

    @field_validator("tools")
    @classmethod
    def _no_duplicate_tools(cls, tools: list[str]) -> list[str]:
        if len(set(tools)) != len(tools):
            raise ValueError("each tool may be listed once")
        return tools

    @model_validator(mode="after")
    def _per_node_caps_fit_inside_the_global_cap(self) -> AgenticConfig:
        """A per-node cap above the global cap is a number that can never bind.

        Left alone, the loop would stop on the global cap while the per-node
        number in the file was never the reason, and the stop reason the caller
        reads would name a limit they did not think they had set.
        """
        caps = self.per_node_llm_calls
        largest = max(caps.plan, caps.assess, caps.repair, caps.generate)
        if largest > self.max_llm_calls:
            raise ValueError(
                f"max_llm_calls of {self.max_llm_calls} is below the largest per-node cap "
                f"of {largest}, which could then never be reached"
            )
        return self

    def node_caps(self) -> dict[NodeName, int]:
        """The per-node caps keyed the way ``AgentState.spend`` reads them."""
        return {
            NodeName.PLAN: self.per_node_llm_calls.plan,
            NodeName.ASSESS: self.per_node_llm_calls.assess,
            NodeName.REPAIR: self.per_node_llm_calls.repair,
            NodeName.GENERATE: self.per_node_llm_calls.generate,
        }


class GraphStrategyConfig(_Strict):
    """Query-time bounds for the graph strategy, both passed to GraphRAGStrategy.

    ``max_hops`` stops at the traversal's own ceiling, because the walk raises
    above it and a config that validates but fails every query is worse than
    one that fails at startup. ``node_budget`` caps the reached set handed to
    generation; the walk's own work is bounded by the hop ceiling, not by it.
    """

    max_hops: int = Field(default=2, ge=1, le=MAX_HOPS_CEILING)
    node_budget: int = Field(default=DEFAULT_NODE_BUDGET, ge=1)


class StrategiesConfig(_Strict):
    traditional: dict[str, float | int | str | bool] = Field(
        default_factory=lambda: {
            "top_k": 8,
            "similarity_threshold": 0.25,
            "rerank": "none",
            "max_context_tokens": 6000,
        }
    )
    vectorless: VectorlessConfig = Field(default_factory=VectorlessConfig)
    agentic: AgenticConfig = Field(default_factory=AgenticConfig)
    graph: GraphStrategyConfig = Field(default_factory=GraphStrategyConfig)


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
