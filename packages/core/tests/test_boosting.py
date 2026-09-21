"""Exact phrase and identifier boosting on top of a lexical ranking."""

import pytest

from ragfabric_core.stores.boosting import apply_boosts, identifiers, phrases
from ragfabric_core.strategies.base import RetrievedChunk


def chunk(chunk_id: int, text: str = "", score: float | None = 1.0) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=1,
        collection_id=1,
        text=text or f"chunk {chunk_id}",
        score=score,
    )


@pytest.mark.parametrize(
    "token,expected",
    [
        # underscore between word characters, with a negative for the branch
        ("ERR_QUOTA_4419", True),
        ("get_user_by_id", True),
        ("getuserbyid", False),
        # a dotted or hyphenated version pattern, with a negative
        ("v2.1.4", True),
        ("1.14.0-rc2", True),
        ("sentence.", False),
        # a digit and a letter in the same token, with a negative
        ("utf8", True),
        ("eight", False),
        # CamelCase with an internal capital, with negatives
        ("CamelCase", True),
        ("retry", False),
        ("limit", False),
        ("The", False),
    ],
)
def test_identifier_detection(token, expected):
    assert (token in identifiers(f"what about {token} here")) is expected


def test_quoted_phrases_are_extracted_verbatim():
    assert phrases('find the "retry limit" now') == ["retry limit"]


def test_an_unclosed_quote_yields_no_phrase_rather_than_the_rest_of_the_query():
    assert phrases('find the "retry limit') == []


def test_a_chunk_containing_the_exact_phrase_is_boosted():
    # Scores are set so the boost has to change the order rather than merely
    # confirm it: without the boost, far (1.0) already outranks near (0.6).
    near = chunk(1, text="the retry limit is 5", score=0.6)
    far = chunk(2, text="retry ... limit", score=1.0)
    out = apply_boosts([far, near], 'the "retry limit"', phrase_boost=2.0, identifier_boost=3.0)
    assert out[0].chunk_id == 1


def test_a_chunk_containing_the_identifier_is_boosted():
    with_id = chunk(1, text="raised as ERR_QUOTA_4419 by the exporter", score=0.5)
    without = chunk(2, text="the exporter raises a quota error", score=1.0)
    out = apply_boosts([without, with_id], "why ERR_QUOTA_4419", 2.0, 3.0)
    assert out[0].chunk_id == 1


def test_boosting_never_invents_a_hit_that_was_not_retrieved():
    assert apply_boosts([], "ERR_QUOTA_4419", 2.0, 3.0) == []


def test_boosting_only_reorders_the_hits_it_was_given():
    """ADR 0003: membership was decided by a store query that ran the access
    filter inside the SQL. A boost that could add a chunk would bypass it."""
    given = [chunk(1, text="no match here"), chunk(2, text="the retry limit is 5")]
    out = apply_boosts(given, '"retry limit" ERR_QUOTA_4419', 2.0, 3.0)
    assert {h.chunk_id for h in out} == {1, 2}
    assert len(out) == 2


def test_an_unmatched_chunk_keeps_its_score_unchanged():
    out = apply_boosts([chunk(1, text="nothing relevant", score=0.42)], '"retry limit"', 2.0, 3.0)
    assert out[0].score == 0.42


def test_a_hit_with_no_score_cannot_be_boosted_and_says_so_by_staying_none():
    out = apply_boosts(
        [chunk(1, text="the retry limit is 5", score=None)], '"retry limit"', 2.0, 3.0
    )
    assert out[0].score is None
