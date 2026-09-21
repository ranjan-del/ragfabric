"""The optional in-process BM25 store, and the cap that keeps it honest."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ragfabric_core.auth.principal import AccessFilter
from ragfabric_core.models import Base
from ragfabric_core.models.document import Chunk, Collection, Document
from ragfabric_core.stores.bm25_memory import InMemoryBm25Store, StoreCapacityError
from ragfabric_core.stores.bm25_sql import Bm25Store

pytest.importorskip("rank_bm25")

ALL = AccessFilter.unrestricted()

CORPUS = [
    "Error ERR_QUOTA_4419 is returned once the retry limit has been exhausted.",
    "The retry limit for the export endpoint is five attempts.",
    "A retry limit protects downstream services from a thundering herd.",
    "Raising the retry limit above ten is discouraged in production.",
    "ERR_QUOTA_4419 also appears in the audit log without a retry.",
    "The quota dashboard lags the enforcement path by a minute.",
]


@pytest.fixture()
def payloads():
    return [{"document_id": 1, "collection_id": 1} for _ in range(3)]


@pytest.fixture()
def store_with_cap_of_2():
    return InMemoryBm25Store(max_chunks=2)


@pytest.fixture()
def sqlite_corpus(tmp_path):
    """A SQLite-backed corpus, indexed identically into both stores.

    SQLite on purpose. Both stores then tokenise with the same portable
    tokeniser, so the comparison is between two implementations of one
    formula. On PostgreSQL the SQL store ranks over to_tsvector's stemmed,
    stop-word-stripped lexemes while the in-process store sees raw tokens, so
    the two index different vocabularies and a disagreement would say nothing
    about either formula.
    """
    engine = create_engine(f"sqlite:///{tmp_path / 'mem.db'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with factory() as db:
        collection = Collection(name="open")
        db.add(collection)
        db.flush()
        document = Document(
            filename="e.txt", format="txt", collection_id=collection.id, status="ready"
        )
        db.add(document)
        db.flush()
        chunks = [
            Chunk(
                document_id=document.id,
                collection_id=collection.id,
                chunk_index=i,
                page=1,
                char_start=0,
                char_end=len(t),
                text=t,
                embedding=[],
            )
            for i, t in enumerate(CORPUS)
        ]
        db.add_all(chunks)
        db.commit()
        ids = [c.id for c in chunks]
        payloads = [{"document_id": document.id, "collection_id": collection.id} for _ in ids]
    return factory, ids, payloads


@pytest.fixture()
def sql_store(sqlite_corpus):
    factory, ids, payloads = sqlite_corpus
    store = Bm25Store(factory)
    store.index(ids, CORPUS, payloads)
    return store


@pytest.fixture()
def mem_store(sqlite_corpus):
    _, ids, payloads = sqlite_corpus
    store = InMemoryBm25Store()
    store.index(ids, CORPUS, payloads)
    return store


def test_exceeding_the_cap_raises_rather_than_degrading(store_with_cap_of_2, payloads):
    with pytest.raises(StoreCapacityError) as exc:
        store_with_cap_of_2.index([1, 2, 3], ["a", "b", "c"], payloads)
    assert "2" in str(exc.value) and "limit" in str(exc.value).lower()


def test_the_cap_error_names_the_alternative(store_with_cap_of_2, payloads):
    with pytest.raises(StoreCapacityError) as exc:
        store_with_cap_of_2.index([1, 2, 3], ["a", "b", "c"], payloads)
    assert "bm25" in str(exc.value)


def test_the_cap_counts_the_corpus_not_one_call(payloads):
    store = InMemoryBm25Store(max_chunks=2)
    store.index([1], ["a"], payloads[:1])
    store.index([2], ["b"], payloads[:1])
    with pytest.raises(StoreCapacityError):
        store.index([3], ["c"], payloads[:1])


def test_a_refused_index_leaves_the_corpus_untouched(payloads):
    store = InMemoryBm25Store(max_chunks=2)
    store.index([1], ["quota"], payloads[:1])
    with pytest.raises(StoreCapacityError):
        store.index([2, 3], ["quota", "quota"], payloads[:2])
    assert store.count() == 1


def test_it_agrees_with_the_sql_implementation_on_a_small_corpus(mem_store, sql_store):
    q = "ERR_QUOTA_4419 retry limit"
    assert [h.chunk_id for h in mem_store.search(q, 5, ALL)] == [
        h.chunk_id for h in sql_store.search(q, 5, ALL)
    ]


def test_the_two_implementations_agree_on_the_scores_too(mem_store, sql_store):
    """Ordering can agree by luck on a small corpus; the numbers cannot."""
    q = "ERR_QUOTA_4419 retry limit"
    mine = {h.chunk_id: h.score for h in mem_store.search(q, 10, ALL)}
    theirs = {h.chunk_id: h.score for h in sql_store.search(q, 10, ALL)}
    assert mine.keys() == theirs.keys()
    for chunk_id, score in mine.items():
        assert score == pytest.approx(theirs[chunk_id], rel=1e-9)


def test_the_access_filter_runs_before_ranking_not_after(mem_store, sqlite_corpus):
    _, _, payloads = sqlite_corpus
    denied = AccessFilter(denied_document_ids=frozenset({payloads[0]["document_id"]}))
    assert mem_store.search("retry limit", 5, denied) == []


def test_an_unknown_filter_key_is_rejected_not_ignored(mem_store):
    with pytest.raises(ValueError):
        mem_store.search("quota", 5, ALL, filters={"nonsense": 1})


def test_deleting_a_document_removes_its_chunks(mem_store, sqlite_corpus):
    _, _, payloads = sqlite_corpus
    mem_store.delete_document(payloads[0]["document_id"])
    assert mem_store.count() == 0
    assert mem_store.search("retry limit", 5, ALL) == []


def test_it_satisfies_the_lexical_store_protocol():
    from ragfabric_core.stores.base import LexicalStore

    assert isinstance(InMemoryBm25Store(), LexicalStore)
