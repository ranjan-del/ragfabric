"""BM25: the pure scoring function, then the store that computes it over the index.

The pure function is tested on its own because it is the piece a reviewer can
check against the formula by eye, and the piece both the SQL store and the
in-process store (Task 8) have to agree with.
"""

import os
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker

from ragfabric_core.auth.principal import AccessFilter
from ragfabric_core.db.migrate import downgrade, upgrade
from ragfabric_core.models import Base
from ragfabric_core.models.document import Chunk, Collection, Document
from ragfabric_core.models.index import ChunkSearch
from ragfabric_core.stores.bm25_sql import Bm25Store, bm25_score
from ragfabric_core.stores.postgres_fts import token_list


def test_idf_is_never_negative_for_a_very_common_term():
    # present in 99 of 100 chunks
    assert bm25_score(tf=1, df=99, n=100, doc_len=10, avgdl=10, k1=1.2, b=0.75) > 0


def test_a_rarer_term_scores_higher_than_a_common_one():
    rare = bm25_score(tf=1, df=1, n=1000, doc_len=10, avgdl=10, k1=1.2, b=0.75)
    common = bm25_score(tf=1, df=900, n=1000, doc_len=10, avgdl=10, k1=1.2, b=0.75)
    assert rare > common


def test_term_frequency_saturates():
    # going 1 -> 2 must help more than 9 -> 10
    first = bm25_score(tf=2, df=5, n=100, doc_len=10, avgdl=10, k1=1.2, b=0.75) - bm25_score(
        tf=1, df=5, n=100, doc_len=10, avgdl=10, k1=1.2, b=0.75
    )
    later = bm25_score(tf=10, df=5, n=100, doc_len=10, avgdl=10, k1=1.2, b=0.75) - bm25_score(
        tf=9, df=5, n=100, doc_len=10, avgdl=10, k1=1.2, b=0.75
    )
    assert first > later


def test_b_zero_disables_length_normalisation():
    short = bm25_score(tf=1, df=5, n=100, doc_len=5, avgdl=100, k1=1.2, b=0.0)
    long = bm25_score(tf=1, df=5, n=100, doc_len=500, avgdl=100, k1=1.2, b=0.0)
    assert short == long


def test_a_long_chunk_is_penalised_when_b_is_on():
    short = bm25_score(tf=1, df=5, n=100, doc_len=5, avgdl=100, k1=1.2, b=0.75)
    long = bm25_score(tf=1, df=5, n=100, doc_len=500, avgdl=100, k1=1.2, b=0.75)
    assert short > long


# ---------------------------------------------------------------------------
# The store, on both dialects it supports.
#
# Parametrised rather than split into two files because the two branches are
# meant to agree: a formula that is right in SQL and wrong in Python (or the
# reverse) is exactly the defect this catches. The PostgreSQL parameter skips
# without RAGFABRIC_TEST_DATABASE_URL, and a skip is not a pass.
# ---------------------------------------------------------------------------

URL = os.environ.get("RAGFABRIC_TEST_DATABASE_URL", "")

ALL = AccessFilter.unrestricted()

IDENTIFIER_CHUNK = (
    "Error ERR_QUOTA_4419 is returned once the retry limit for the export "
    "endpoint has been exhausted."
)
COMMON_CHUNKS = [
    "The retry limit for the export endpoint is five attempts.",
    "A retry limit protects downstream services from a thundering herd.",
    "Raising the retry limit above ten is discouraged in production.",
    "Each connector has its own retry limit and its own backoff curve.",
    "The retry limit is configured per environment, not per request.",
]


@pytest.fixture(params=["sqlite", "postgresql"])
def corpus(request, tmp_path):
    if request.param == "postgresql":
        if not URL:
            pytest.skip("needs RAGFABRIC_TEST_DATABASE_URL")
        downgrade(URL)
        upgrade(URL)
        engine = create_engine(URL)
    else:
        engine = create_engine(f"sqlite:///{tmp_path / 'bm25.db'}")
        Base.metadata.create_all(engine)

    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with factory() as db:
        open_collection, other_collection = Collection(name="open"), Collection(name="other")
        db.add_all([open_collection, other_collection])
        db.flush()
        document = Document(
            filename="errors.txt",
            format="txt",
            collection_id=open_collection.id,
            status="ready",
        )
        other_document = Document(
            filename="notes.md",
            format="md",
            collection_id=other_collection.id,
            status="ready",
        )
        db.add_all([document, other_document])
        db.flush()

        texts = [IDENTIFIER_CHUNK, *COMMON_CHUNKS]
        chunks = [
            Chunk(
                document_id=document.id,
                collection_id=open_collection.id,
                chunk_index=i,
                page=1,
                char_start=0,
                char_end=len(t),
                text=t,
                embedding=[],
            )
            for i, t in enumerate(texts)
        ]
        db.add_all(chunks)
        db.commit()
        ids = [c.id for c in chunks]
        collection_ids = (open_collection.id, other_collection.id)
        document_ids = (document.id, other_document.id)

    store = Bm25Store(factory)
    store.index(
        ids,
        texts,
        [{"document_id": document_ids[0], "collection_id": collection_ids[0]} for _ in ids],
    )
    return SimpleNamespace(
        factory=factory,
        store=store,
        dialect=request.param,
        chunk_ids=ids,
        chunk_with_error_code=ids[0],
        collection_ids=collection_ids,
        document_ids=document_ids,
    )


@pytest.fixture()
def store(corpus):
    return corpus.store


@pytest.fixture()
def legacy_row(corpus):
    """A chunk_search row as Phase 3 wrote them: no doc_len, a set based tsv.

    Written straight to the table on purpose. Going through the store would
    populate doc_len and the term statistics, which is precisely the state
    this fixture must not be in.
    """
    text_value = "retry limit ERR_QUOTA_4419 retry limit"
    with corpus.factory() as db:
        chunk = Chunk(
            document_id=corpus.document_ids[0],
            collection_id=corpus.collection_ids[0],
            chunk_index=99,
            page=1,
            char_start=0,
            char_end=len(text_value),
            text=text_value,
            embedding=[],
        )
        db.add(chunk)
        db.flush()
        tsv = (
            func.to_tsvector("english", text_value)
            if db.bind.dialect.name == "postgresql"
            else " ".join(sorted(set(token_list(text_value))))
        )
        db.add(
            ChunkSearch(
                chunk_id=chunk.id,
                document_id=corpus.document_ids[0],
                collection_id=corpus.collection_ids[0],
                tsv=tsv,
                doc_len=0,
            )
        )
        db.commit()
        return SimpleNamespace(chunk_id=chunk.id)


def test_an_identifier_outranks_a_common_word(store, corpus):
    # corpus: 'ERR_QUOTA_4419' in one chunk, 'limit' in most
    hits = store.search("what is the retry limit for ERR_QUOTA_4419", top_k=3, access=ALL)
    assert hits[0].chunk_id == corpus.chunk_with_error_code


def test_doc_len_zero_rows_are_excluded_not_scored(store, legacy_row):
    # The query is one the legacy row matches strongly, so this fails if the
    # row is scored at all rather than merely failing to reach the top.
    hits = store.search("retry limit ERR_QUOTA_4419", 10, ALL)
    assert legacy_row.chunk_id not in {h.chunk_id for h in hits}


def test_the_access_filter_runs_inside_the_query(store, corpus):
    denied = AccessFilter(denied_document_ids=frozenset({corpus.document_ids[0]}))
    assert store.search("retry limit", top_k=10, access=denied) == []


def test_repetition_raises_the_score_so_term_frequency_survived(corpus):
    """tf must reach the formula on both dialects.

    Before Phase 4 the off-PostgreSQL column stored a set, so a term repeated
    nine times looked identical to one occurring once and this could not be
    true.
    """
    with corpus.factory() as db:
        chunk = Chunk(
            document_id=corpus.document_ids[0],
            collection_id=corpus.collection_ids[0],
            chunk_index=50,
            page=1,
            char_start=0,
            char_end=1,
            text="ERR_QUOTA_4419 " * 6,
            embedding=[],
        )
        db.add(chunk)
        db.commit()
        repeated_id = chunk.id
    corpus.store.index(
        [repeated_id],
        ["ERR_QUOTA_4419 " * 6],
        [{"document_id": corpus.document_ids[0], "collection_id": corpus.collection_ids[0]}],
    )
    hits = corpus.store.search("ERR_QUOTA_4419", top_k=5, access=ALL)
    by_id = {h.chunk_id: h.score for h in hits}
    assert by_id[repeated_id] > by_id[corpus.chunk_with_error_code]
