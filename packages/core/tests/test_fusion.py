"""Reciprocal Rank Fusion over ranked lists of chunks."""

from ragfabric_core.stores.fusion import rrf
from ragfabric_core.strategies.base import RetrievedChunk


def chunk(chunk_id: int, score: float | None = None, text: str = "") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=1,
        collection_id=1,
        text=text or f"chunk {chunk_id}",
        score=score,
    )


def test_a_chunk_ranked_first_by_both_wins():
    a, b = chunk(1), chunk(2)
    assert rrf([[a, b], [a, b]])[0].chunk_id == 1


def test_fusion_promotes_a_chunk_both_rank_second_over_a_one_list_winner():
    # a is 1st in one list and absent from the other; b is 2nd in both
    a, b, c = chunk(1), chunk(2), chunk(3)
    out = rrf([[a, b], [c, b]])
    assert out[0].chunk_id == 2


def test_missing_from_a_ranking_contributes_nothing_rather_than_a_penalty():
    a, b = chunk(1), chunk(2)
    assert {h.chunk_id for h in rrf([[a], [b]])} == {1, 2}


def test_weights_shift_the_outcome():
    a, b = chunk(1), chunk(2)
    assert rrf([[a, b], [b, a]], weights=[10.0, 1.0])[0].chunk_id == 1


def test_the_result_carries_the_fused_score_not_an_input_score():
    a = chunk(1, score=999.0)
    assert rrf([[a]])[0].score != 999.0


def test_ties_break_on_chunk_id_so_the_order_is_deterministic():
    a, b = chunk(7), chunk(3)
    assert [h.chunk_id for h in rrf([[a, b], [b, a]])] == [3, 7]


def test_fusion_introduces_no_chunk_that_was_not_in_a_ranking():
    """ADR 0003: membership is decided by retrieval, which ran the access filter.

    Fusion reorders. If it could add a chunk, it would be adding one no store
    was willing to return for this principal.
    """
    a, b = chunk(1), chunk(2)
    assert {h.chunk_id for h in rrf([[a], [a, b]])} == {1, 2}
    assert rrf([[], []]) == []
