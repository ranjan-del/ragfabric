"""Maintenance of the BM25 term and corpus statistics.

These statistics drift silently when they are wrong: a bad df does not raise,
it just ranks badly forever. So the invariants are pinned here rather than
left to the store that calls them.
"""

import os

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ragfabric_core.db.migrate import downgrade, upgrade
from ragfabric_core.models import Base
from ragfabric_core.models.index import CorpusStat, TermStat
from ragfabric_core.stores.term_stats import corpus_stats, record_indexed, record_removed


@pytest.fixture()
def session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'term_stats.db'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    with factory() as db:
        yield db


def test_df_counts_chunks_not_occurrences(session):
    record_indexed(session, [{"quota": 9}])
    assert session.get(TermStat, "quota").df == 1


def test_reindexing_a_chunk_does_not_inflate_df(session):
    record_indexed(session, [{"quota": 1}])
    record_removed(session, [{"quota": 1}])
    record_indexed(session, [{"quota": 1}])
    assert session.get(TermStat, "quota").df == 1


def test_df_accumulates_across_separately_indexed_chunks(session):
    """The guard for ``df = df + delta`` being arithmetic rather than assignment.

    The plan nominates test_reindexing_a_chunk_does_not_inflate_df for this,
    but that sequence survives an assigning implementation: the removal drops
    df to zero, the row is deleted, and the re-index puts back exactly 1
    either way. Two chunks indexed in separate calls is the sequence that
    actually distinguishes the two.
    """
    record_indexed(session, [{"quota": 1}])
    record_indexed(session, [{"quota": 1}])
    assert session.get(TermStat, "quota").df == 2


def test_df_reaching_zero_removes_the_row(session):
    record_indexed(session, [{"gone": 1}])
    record_removed(session, [{"gone": 1}])
    assert session.get(TermStat, "gone") is None


def test_avgdl_is_one_on_an_empty_corpus(session):
    assert corpus_stats(session) == (0, 1.0)


def test_corpus_totals_track_chunks_and_term_occurrences(session):
    record_indexed(session, [{"quota": 9, "retry": 1}, {"quota": 2, "limit": 8}])
    stats = session.get(CorpusStat, 1)
    assert stats.n_chunks == 2
    assert stats.sum_len == 20
    assert corpus_stats(session) == (2, 10.0)


def test_removing_every_chunk_returns_the_corpus_to_empty(session):
    chunks = [{"quota": 3, "retry": 1}]
    record_indexed(session, chunks)
    record_removed(session, chunks)
    assert corpus_stats(session) == (0, 1.0)
    assert session.get(TermStat, "quota") is None


def test_a_term_longer_than_the_key_width_is_truncated_not_rejected(session):
    long_term = "z" * 400
    record_indexed(session, [{long_term: 1}])
    assert session.get(TermStat, long_term[:255]).df == 1


URL = os.environ.get("RAGFABRIC_TEST_DATABASE_URL", "")


@pytest.fixture()
def pg_session():
    downgrade(URL)
    upgrade(URL)
    engine = create_engine(URL)
    factory = sessionmaker(bind=engine)
    with factory() as db:
        yield db


@pytest.mark.integration
@pytest.mark.skipif(not URL, reason="needs RAGFABRIC_TEST_DATABASE_URL")
def test_the_same_invariants_hold_on_real_postgresql(pg_session):
    """The upsert is dialect specific, so the PostgreSQL branch gets its own run.

    SQLite and PostgreSQL take different code paths through _upsert_df, and a
    branch that only ever runs behind a skipped guard is a branch nobody has
    tested. Migration 0007 has already seeded corpus_stats id=1 here, so this
    also covers the UPDATE path that the metadata-built SQLite schema skips.
    """
    record_indexed(pg_session, [{"quota": 9, "retry": 1}])
    record_indexed(pg_session, [{"quota": 1}])
    assert pg_session.get(TermStat, "quota").df == 2
    assert corpus_stats(pg_session) == (2, 5.5)

    record_removed(pg_session, [{"quota": 1}])
    assert pg_session.get(TermStat, "quota").df == 1

    record_removed(pg_session, [{"quota": 9, "retry": 1}])
    assert pg_session.get(TermStat, "quota") is None
    assert pg_session.get(TermStat, "retry") is None
    assert corpus_stats(pg_session) == (0, 1.0)
