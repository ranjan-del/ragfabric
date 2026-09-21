"""Model tests for the BM25 term statistics tables.

term_stats.df (document frequency per term) and corpus_stats (the running
totals behind avgdl) are new in Phase 4. This exercises them on SQLite, the
dialect the rest of the unit suite runs against; PostgreSQL specific
behaviour is covered by the migration tests.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ragfabric_core.models import Base
from ragfabric_core.models.index import CorpusStat, TermStat


@pytest.fixture()
def session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'term_stats.db'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    with factory() as db:
        yield db


def test_term_stat_and_corpus_stat_round_trip(session):
    session.add(TermStat(term="quota", df=3))
    session.add(CorpusStat(id=1, n_chunks=10, sum_len=6000))
    session.commit()
    assert session.get(TermStat, "quota").df == 3
    stats = session.get(CorpusStat, 1)
    assert stats.n_chunks == 10 and stats.sum_len == 6000
