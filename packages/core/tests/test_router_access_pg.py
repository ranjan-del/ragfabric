"""No denied chunk escapes through any auto path, on real PostgreSQL stores (ADR 0003).

The offline suite proves the router's branches with fakes. This module drives
the real ``AutoStrategy`` built by ``default_registry`` over real pgvector and
lexical stores, with the offline hashing embedder and a scripted model, so it
needs ``RAGFABRIC_TEST_DATABASE_URL`` and nothing else. Two collections are
seeded and the caller may read only one. Each path that can hand a request to
a second strategy is driven once:

- a decisive Vectorless pick that finds nothing visible, so Traditional answers;
- a decisive agentic pick whose model raises, so the error fallback answers;
- an unclear question the classifier is not sure of, so the two searches fuse.

Every path must keep the caller's filter: no chunk of the denied collection
may appear in the chunks, in a sub-question's evidence, or anywhere the trace
reports.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ragfabric_core.auth.principal import AccessFilter, Principal
from ragfabric_core.config_file import RagFabricConfig
from ragfabric_core.db.migrate import downgrade, upgrade
from ragfabric_core.models.document import Chunk, Collection, Document
from ragfabric_core.providers.base import Completion, Message, ProviderError
from ragfabric_core.providers.registry import build_embedding_provider
from ragfabric_core.stores.pgvector_store import PgVectorStore
from ragfabric_core.stores.postgres_fts import PostgresLexicalStore
from ragfabric_core.strategies import registry_defaults
from ragfabric_core.strategies.base import (
    RetrievalContext,
    RetrievalResult,
    StrategyName,
    StrategyParams,
)
from ragfabric_core.strategies.contract import assert_strategy_contract
from ragfabric_core.workers.handlers import index_document

URL = os.environ.get("RAGFABRIC_TEST_DATABASE_URL", "")

pytestmark = pytest.mark.skipif(not URL, reason="needs RAGFABRIC_TEST_DATABASE_URL")

SECRET_CODE = "ERR_VAULT_9001"
OPEN_CORPUS = {
    "handbook.txt": [
        "Full time employees accrue twenty two days of annual leave per calendar year.",
        "The office opens at nine in the morning and closes at six in the evening.",
        "Expense claims are submitted through the finance portal within thirty days.",
    ],
}
DENIED_CORPUS = {
    "vault.txt": [
        f"{SECRET_CODE} means the payroll vault key was rotated by the security team.",
        "Executives accrue forty days of annual leave per calendar year.",
        "The payroll vault is audited every quarter by the security team.",
    ],
}
IDENTIFIER_QUESTION = f"What does {SECRET_CODE} mean?"
AGGREGATION_QUESTION = "How many days of annual leave do employees accrue?"
# Longer than the plain limit and firing no rule, so the classifier decides.
OPEN_QUESTION = (
    "could you explain what people here generally get in terms of annual leave "
    "and time off across a normal calendar year for someone in the company"
)


class ScriptedModel:
    """One model for every strategy: replies from a script, or raises when told to."""

    name = "scripted"
    default_model = "scripted-model"

    def __init__(self) -> None:
        self.replies: list[str] = []
        self.fail = False
        self.calls = 0

    def complete(self, messages: list[Message], **kwargs) -> Completion:
        self.calls += 1
        if self.fail:
            raise ProviderError(self.name, "model unreachable")
        if not self.replies:
            raise ProviderError(self.name, "script exhausted")
        text = self.replies.pop(0)
        return Completion(
            text=text,
            model=self.default_model,
            provider=self.name,
            input_tokens=1,
            output_tokens=1,
            latency_ms=0,
        )

    def stream(self, messages, **kwargs):
        yield from ()


@dataclass
class Seeded:
    factory: sessionmaker
    open_collection_id: int
    denied_chunk_ids: set[int]


def _config() -> RagFabricConfig:
    return RagFabricConfig.model_validate(
        {
            "llm": {"provider": "offline"},
            "embeddings": {"provider": "offline", "dim": 768},
        }
    )


@pytest.fixture()
def seeded():
    downgrade(URL)
    upgrade(URL)
    engine = create_engine(URL)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    embedder = build_embedding_provider(_config().embeddings)
    vector_store = PgVectorStore(factory, model=embedder.model)
    lexical_store = PostgresLexicalStore(factory)

    with factory() as db:
        ids: dict[str, int] = {}
        document_ids: list[int] = []
        for label, corpus in (("open", OPEN_CORPUS), ("denied", DENIED_CORPUS)):
            collection = Collection(name=f"router-access-{label}")
            db.add(collection)
            db.flush()
            ids[label] = collection.id
            for filename, texts in corpus.items():
                document = Document(
                    filename=filename,
                    format="txt",
                    collection_id=collection.id,
                    status="processing",
                )
                db.add(document)
                db.flush()
                document_ids.append(document.id)
                for index, text in enumerate(texts):
                    db.add(
                        Chunk(
                            document_id=document.id,
                            collection_id=collection.id,
                            chunk_index=index,
                            page=1,
                            char_start=0,
                            char_end=len(text),
                            text=text,
                            embedding=[],
                        )
                    )
        db.commit()
        for document_id in document_ids:
            index_document(
                db,
                document_id,
                embedding_provider=embedder,
                vector_store=vector_store,
                lexical_store=lexical_store,
            )
        denied = {
            row.id for row in db.query(Chunk).filter(Chunk.collection_id == ids["denied"]).all()
        }

    yield Seeded(factory=factory, open_collection_id=ids["open"], denied_chunk_ids=denied)
    engine.dispose()
    downgrade(URL)


@pytest.fixture()
def auto(seeded, monkeypatch):
    model = ScriptedModel()
    monkeypatch.setattr(registry_defaults, "build_llm_provider", lambda cfg: model)
    registry = registry_defaults.default_registry(_config(), seeded.factory)
    return registry.get(StrategyName.AUTO), model


def _ctx(access: AccessFilter) -> RetrievalContext:
    return RetrievalContext(
        principal=Principal(user_id=1, email="engineer@example.com"),
        access_filter=access,
        params=StrategyParams(top_k=5),
    )


def _restricted(seeded: Seeded) -> RetrievalContext:
    return _ctx(AccessFilter(collection_ids=frozenset({seeded.open_collection_id})))


def _reported_ids(result: RetrievalResult) -> set[int]:
    """Every chunk id the result names: its chunks, its sub-questions and its trace."""
    found = {chunk.chunk_id for chunk in result.chunks}
    for report in result.sub_questions:
        found.update(report.chunk_ids)
    if result.subgraph is not None:
        for edge in result.subgraph.edges:
            found.update(edge.source_chunk_ids)
    for span in result.trace:
        for key, value in span.attributes.items():
            if "chunk" not in key:
                continue
            values = value if isinstance(value, list | tuple | set) else [value]
            found.update(v for v in values if isinstance(v, int) and not isinstance(v, bool))
    return found


def _assert_nothing_denied(result: RetrievalResult, seeded: Seeded) -> None:
    assert not (_reported_ids(result) & seeded.denied_chunk_ids)
    assert all(chunk.collection_id == seeded.open_collection_id for chunk in result.chunks)
    trace_text = json.dumps([span.attributes for span in result.trace], default=str)
    assert SECRET_CODE not in trace_text and "payroll vault" not in trace_text


def test_the_seed_would_leak_without_the_filter(seeded, auto):
    # The control: unrestricted, the identifier question finds the denied chunk,
    # so the restricted runs below prove the filter held, not an empty corpus.
    strategy, _ = auto
    result = strategy.retrieve(IDENTIFIER_QUESTION, _ctx(AccessFilter.unrestricted()))
    assert result.strategy is StrategyName.VECTORLESS
    assert {chunk.chunk_id for chunk in result.chunks} & seeded.denied_chunk_ids


def test_an_empty_vectorless_pick_falls_back_without_leaking(seeded, auto):
    strategy, model = auto
    result = assert_strategy_contract(strategy, IDENTIFIER_QUESTION, _restricted(seeded))
    assert result.router.selected_strategy == "vectorless" and result.router.decisive
    assert result.fallback_from is StrategyName.VECTORLESS
    assert result.strategy is StrategyName.TRADITIONAL
    assert result.chunks  # Traditional found open chunks
    assert model.calls == 0
    _assert_nothing_denied(result, seeded)


def test_an_agentic_pick_whose_model_raises_falls_back_without_leaking(seeded, auto):
    strategy, model = auto
    model.fail = True
    result = assert_strategy_contract(strategy, AGGREGATION_QUESTION, _restricted(seeded))
    assert result.router.selected_strategy == "agentic" and result.router.decisive
    assert result.fallback_from is StrategyName.AGENTIC
    assert result.strategy is StrategyName.TRADITIONAL
    router_span = next(span for span in result.trace if span.name == "router")
    assert router_span.attributes.get("fallback_reason")
    assert model.calls >= 1
    _assert_nothing_denied(result, seeded)


def test_the_fused_path_keeps_the_filter(seeded, auto):
    strategy, model = auto
    model.replies = [
        json.dumps(
            {
                "query_type": "multi_hop",
                "strategy": "agentic",
                "confidence": 0.1,
                "reasoning": "Not sure.",
            }
        )
    ]
    result = assert_strategy_contract(strategy, OPEN_QUESTION, _restricted(seeded))
    assert result.router.source == "classifier" and result.router.fused is True
    assert result.chunks
    assert model.calls == 1
    _assert_nothing_denied(result, seeded)
