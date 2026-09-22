"""The three agent tools, and the one property that matters most about them.

Tools are thin wrappers over strategies that are already tested elsewhere, so
these tests do not re-test retrieval. They test the two things wrapping can get
wrong: the name and description the planner sees, and whether the caller's
``AccessFilter`` survives the trip.

``fetch_document`` gets the most attention because it is the one tool that
reaches a document without searching for it. If it does not apply the same
predicate a search applies, the agent becomes a way to read documents the
principal is not allowed to find, which is exactly the hole ADR 0003 exists to
close. That is proved twice: once that the tool hands the filter to its reader,
and once that the real SQL reader puts the predicate inside the query rather
than filtering afterwards in Python.
"""

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from ragfabric_core.agent.tools import (
    AgentTool,
    FetchDocumentTool,
    LexicalSearchTool,
    SemanticSearchTool,
    build_tool_registry,
)
from ragfabric_core.auth.principal import AccessFilter, Principal
from ragfabric_core.models import Base
from ragfabric_core.models.document import Chunk, Collection, Document
from ragfabric_core.stores.document_chunks import SqlDocumentChunkReader
from ragfabric_core.strategies.base import (
    RetrievalContext,
    RetrievedChunk,
    StrategyName,
    StrategyParams,
)
from ragfabric_core.strategies.traditional import TraditionalRAGStrategy
from ragfabric_core.strategies.vectorless import VectorlessRAGStrategy

OPEN_COLLECTION = 1
SECRET_COLLECTION = 2
OPEN_DOCUMENT = 101
SECRET_DOCUMENT = 102


def chunk(chunk_id: int, *, document_id: int = OPEN_DOCUMENT, collection_id: int = OPEN_COLLECTION):
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        collection_id=collection_id,
        text=f"passage {chunk_id} about the retry limit",
        score=1.0,
    )


class StubVectorStore:
    """A VectorStore double that honours the filter, as a real one must."""

    name = "stub_vector"

    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self._chunks = chunks
        self.last_access: AccessFilter | None = None

    def upsert(self, chunk_ids, vectors, payloads):  # pragma: no cover
        raise NotImplementedError

    def delete_document(self, document_id):  # pragma: no cover
        raise NotImplementedError

    def count(self):  # pragma: no cover
        raise NotImplementedError

    def access_stats(self, filters, access):  # pragma: no cover
        raise NotImplementedError

    def query(self, vector, top_k, access, filters=None):
        self.last_access = access
        return [c for c in self._chunks if access.allows(c.document_id, c.collection_id)][:top_k]


class StubLexicalStore:
    def __init__(self, name: str, chunks: list[RetrievedChunk]) -> None:
        self.name = name
        self._chunks = chunks
        self.last_access: AccessFilter | None = None

    def index(self, chunk_ids, texts, payloads):  # pragma: no cover
        raise NotImplementedError

    def delete_document(self, document_id):  # pragma: no cover
        raise NotImplementedError

    def search(self, query, top_k, access, filters=None):
        self.last_access = access
        return [c for c in self._chunks if access.allows(c.document_id, c.collection_id)][:top_k]


class StubEmbedder:
    name = "stub"
    model = "stub-embed"
    dim = 3

    def embed(self, texts):
        from ragfabric_core.providers.base import EmbeddingResult

        return EmbeddingResult(
            vectors=[[1.0, 0.0, 0.0] for _ in texts],
            model=self.model,
            provider=self.name,
            input_tokens=len(texts),
            latency_ms=0,
        )


class StubChunkReader:
    """A DocumentChunkReader double. It applies the filter it is handed."""

    name = "stub_reader"

    def __init__(self, by_document: dict[int, list[RetrievedChunk]]) -> None:
        self._by_document = by_document
        self.last_access: AccessFilter | None = None
        self.last_document_id: int | None = None

    def chunks_for_document(self, document_id: int, access: AccessFilter):
        self.last_access = access
        self.last_document_id = document_id
        return [
            c
            for c in self._by_document.get(document_id, [])
            if access.allows(c.document_id, c.collection_id)
        ]


def ctx(access: AccessFilter | None = None, top_k: int = 5) -> RetrievalContext:
    return RetrievalContext(
        principal=Principal(user_id=1, email="a@b.c", role="user"),
        access_filter=access or AccessFilter.unrestricted(),
        params=StrategyParams(top_k=top_k),
    )


def semantic_tool(chunks: list[RetrievedChunk]) -> tuple[SemanticSearchTool, StubVectorStore]:
    store = StubVectorStore(chunks)
    strategy = TraditionalRAGStrategy(embedding_provider=StubEmbedder(), vector_store=store)
    return SemanticSearchTool(strategy), store


def lexical_tool(
    chunks: list[RetrievedChunk],
) -> tuple[LexicalSearchTool, StubLexicalStore, StubLexicalStore]:
    bm25 = StubLexicalStore("bm25", chunks)
    ts_rank = StubLexicalStore("postgres_fts", chunks)
    strategy = VectorlessRAGStrategy(bm25_store=bm25, ts_rank_store=ts_rank)
    return LexicalSearchTool(strategy), bm25, ts_rank


def test_the_tools_are_named_exactly_what_the_planner_is_told() -> None:
    tool_semantic, _ = semantic_tool([])
    tool_lexical, _, _ = lexical_tool([])
    tool_fetch = FetchDocumentTool(StubChunkReader({}))
    assert tool_semantic.name == "semantic_search"
    assert tool_lexical.name == "lexical_search"
    assert tool_fetch.name == "fetch_document"


def test_each_description_tells_the_planner_what_the_tool_is_good_at() -> None:
    tool_semantic, _ = semantic_tool([])
    tool_lexical, _, _ = lexical_tool([])
    tool_fetch = FetchDocumentTool(StubChunkReader({}))
    assert "identifier" in tool_lexical.description.lower()
    assert "concept" in tool_semantic.description.lower()
    assert "whole policy" in tool_fetch.description.lower()


def test_every_tool_satisfies_the_agent_tool_protocol() -> None:
    tool_semantic, _ = semantic_tool([])
    tool_lexical, _, _ = lexical_tool([])
    tool_fetch = FetchDocumentTool(StubChunkReader({}))
    for tool in (tool_semantic, tool_lexical, tool_fetch):
        assert isinstance(tool, AgentTool)


def test_semantic_search_returns_what_the_traditional_strategy_retrieved() -> None:
    tool, _ = semantic_tool([chunk(1), chunk(2)])
    got = tool.run("what is the retry limit", ctx())
    assert [c.chunk_id for c in got] == [1, 2]


def test_lexical_search_returns_what_the_vectorless_strategy_retrieved() -> None:
    tool, _, _ = lexical_tool([chunk(1), chunk(2)])
    got = tool.run("ERR_QUOTA_4419", ctx())
    assert {c.chunk_id for c in got} == {1, 2}


def test_each_tool_passes_the_access_filter_through() -> None:
    """The caller's filter object reaches the store, not a copy or a default."""
    access = AccessFilter(collection_ids=frozenset({OPEN_COLLECTION}))
    context = ctx(access=access)

    tool_semantic, vector_store = semantic_tool([chunk(1)])
    tool_semantic.run("retry limit", context)
    assert vector_store.last_access is access

    tool_lexical, bm25, ts_rank = lexical_tool([chunk(1)])
    tool_lexical.run("retry limit", context)
    assert bm25.last_access is access
    assert ts_rank.last_access is access

    reader = StubChunkReader({OPEN_DOCUMENT: [chunk(1)]})
    FetchDocumentTool(reader).run(str(OPEN_DOCUMENT), context)
    assert reader.last_access is access


def test_fetch_document_refuses_a_document_the_principal_cannot_read() -> None:
    """Fetching by id must not be a way round the search access predicate."""
    reader = StubChunkReader(
        {SECRET_DOCUMENT: [chunk(9, document_id=SECRET_DOCUMENT, collection_id=SECRET_COLLECTION)]}
    )
    tool = FetchDocumentTool(reader)
    permitted = ctx(access=AccessFilter(collection_ids=frozenset({OPEN_COLLECTION})))
    assert tool.run(str(SECRET_DOCUMENT), permitted) == []


def test_fetch_document_returns_the_whole_document_in_order() -> None:
    reader = StubChunkReader({OPEN_DOCUMENT: [chunk(3), chunk(1), chunk(2)]})
    tool = FetchDocumentTool(reader)
    assert [c.chunk_id for c in tool.run(str(OPEN_DOCUMENT), ctx())] == [3, 1, 2]


def test_fetch_document_accepts_a_document_id_embedded_in_the_query() -> None:
    """The id arrives from a model, so "document 101" must work as well as "101"."""
    reader = StubChunkReader({OPEN_DOCUMENT: [chunk(1)]})
    tool = FetchDocumentTool(reader)
    assert [c.chunk_id for c in tool.run(f"document {OPEN_DOCUMENT}", ctx())] == [1]


def test_fetch_document_with_no_document_id_retrieves_nothing() -> None:
    """A repair that names no document is a dead end, not a crash."""
    reader = StubChunkReader({OPEN_DOCUMENT: [chunk(1)]})
    tool = FetchDocumentTool(reader)
    assert tool.run("the whole leave policy", ctx()) == []
    assert reader.last_document_id is None


def test_the_registry_resolves_the_three_tools_by_name() -> None:
    tool_semantic, _ = semantic_tool([])
    tool_lexical, _, _ = lexical_tool([])
    tool_fetch = FetchDocumentTool(StubChunkReader({}))
    registry = build_tool_registry([tool_semantic, tool_lexical, tool_fetch])
    assert sorted(registry) == ["fetch_document", "lexical_search", "semantic_search"]
    assert registry["semantic_search"] is tool_semantic


def test_the_registry_keeps_only_the_enabled_tools() -> None:
    tool_semantic, _ = semantic_tool([])
    tool_lexical, _, _ = lexical_tool([])
    tool_fetch = FetchDocumentTool(StubChunkReader({}))
    registry = build_tool_registry(
        [tool_semantic, tool_lexical, tool_fetch], enabled=["semantic_search", "fetch_document"]
    )
    assert sorted(registry) == ["fetch_document", "semantic_search"]


def test_enabling_a_tool_that_does_not_exist_is_an_error() -> None:
    tool_semantic, _ = semantic_tool([])
    with pytest.raises(KeyError):
        build_tool_registry([tool_semantic], enabled=["semantic_serch"])


@pytest.fixture()
def sf(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'tools.db'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with factory() as db:
        open_collection = Collection(name="open")
        secret_collection = Collection(name="secret")
        db.add_all([open_collection, secret_collection])
        db.flush()
        open_document = Document(
            filename="leave.txt", format="txt", collection_id=open_collection.id, status="ready"
        )
        secret_document = Document(
            filename="bonus.txt", format="txt", collection_id=secret_collection.id, status="ready"
        )
        db.add_all([open_document, secret_document])
        db.flush()
        db.add_all(
            [
                Chunk(
                    document_id=open_document.id,
                    collection_id=open_collection.id,
                    chunk_index=index,
                    page=1,
                    char_start=0,
                    char_end=10,
                    text=f"leave policy part {index}",
                    embedding=[],
                )
                # Inserted out of order so ordering by chunk_index is a real assertion
                # rather than an accident of insertion order.
                for index in (2, 0, 1)
            ]
            + [
                Chunk(
                    document_id=secret_document.id,
                    collection_id=secret_collection.id,
                    chunk_index=0,
                    page=1,
                    char_start=0,
                    char_end=10,
                    text="secret bonus rules",
                    embedding=[],
                )
            ]
        )
        db.commit()
        ids = (open_document.id, secret_document.id, open_collection.id, secret_collection.id)
    return factory, ids


def test_the_sql_reader_returns_a_document_in_chunk_order(sf) -> None:
    factory, (open_document, _, _, _) = sf
    reader = SqlDocumentChunkReader(factory)
    got = reader.chunks_for_document(open_document, AccessFilter.unrestricted())
    assert [c.text for c in got] == [
        "leave policy part 0",
        "leave policy part 1",
        "leave policy part 2",
    ]


def test_the_sql_reader_applies_the_access_predicate_inside_the_query(sf) -> None:
    """A denied document comes back empty from the database, not filtered after."""
    factory, (open_document, secret_document, open_collection, _) = sf
    reader = SqlDocumentChunkReader(factory)
    permitted = AccessFilter(collection_ids=frozenset({open_collection}))
    assert reader.chunks_for_document(secret_document, permitted) == []
    assert reader.chunks_for_document(open_document, permitted) != []


def test_the_access_predicate_is_in_the_sql_not_applied_afterwards(sf) -> None:
    """ADR 0003 is about where the predicate runs, so look at the SQL that ran.

    A reader that fetched every chunk and dropped the forbidden ones in Python
    would pass the two tests above and still be the thing ADR 0003 forbids, so
    this one records the statement the database was actually sent.
    """
    factory, (open_document, _, open_collection, _) = sf
    engine = factory.kw["bind"]
    statements: list[str] = []

    @event.listens_for(engine, "before_cursor_execute")
    def record(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)

    try:
        SqlDocumentChunkReader(factory).chunks_for_document(
            open_document, AccessFilter(collection_ids=frozenset({open_collection}))
        )
    finally:
        event.remove(engine, "before_cursor_execute", record)

    assert statements, "no statement reached the database"
    where = statements[-1].lower()
    assert "collection_id in" in where


def test_the_sql_reader_honours_the_deny_list(sf) -> None:
    factory, (open_document, _, _, _) = sf
    reader = SqlDocumentChunkReader(factory)
    denied = AccessFilter(denied_document_ids=frozenset({open_document}))
    assert reader.chunks_for_document(open_document, denied) == []


def test_fetch_document_over_the_sql_reader_refuses_a_denied_document(sf) -> None:
    """End to end on a real database: the tool plus the real reader, no doubles."""
    factory, (_, secret_document, open_collection, _) = sf
    tool = FetchDocumentTool(SqlDocumentChunkReader(factory))
    context = ctx(access=AccessFilter(collection_ids=frozenset({open_collection})))
    assert tool.run(str(secret_document), context) == []


def test_the_traditional_strategy_still_reports_its_own_name() -> None:
    """A tool wraps a strategy, it does not replace or rename it."""
    tool, _ = semantic_tool([])
    assert tool.strategy.name is StrategyName.TRADITIONAL
