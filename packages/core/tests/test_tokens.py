from ragfabric_core.strategies.base import RetrievedChunk
from ragfabric_core.tokens import count_tokens, fit_to_budget


def chunk(cid: int, text: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid, document_id=1, collection_id=None, text=text, score=score
    )


def test_count_tokens_is_exact_for_a_known_openai_model():
    assert count_tokens("hello world", "text-embedding-3-small") == 2


def test_count_tokens_estimates_for_an_unknown_model_without_raising():
    assert count_tokens("a" * 40, "llama3.2:3b") == 10


def test_count_tokens_of_empty_text_is_zero():
    assert count_tokens("", None) == 0


def test_fit_to_budget_keeps_rank_order_and_drops_the_tail():
    chunks = [chunk(1, "a" * 40, 0.9), chunk(2, "b" * 40, 0.8), chunk(3, "c" * 40, 0.7)]
    kept, total = fit_to_budget(chunks, max_tokens=20, model="llama3.2:3b")
    assert [c.chunk_id for c in kept] == [1, 2]
    assert total == 20


def test_fit_to_budget_never_splits_a_chunk():
    chunks = [chunk(1, "a" * 400, 0.9)]
    kept, total = fit_to_budget(chunks, max_tokens=10, model="llama3.2:3b")
    assert kept == [], "a chunk that does not fit is dropped whole, never truncated"
    assert total == 0


def test_fit_to_budget_with_a_generous_budget_keeps_everything():
    chunks = [chunk(1, "a" * 40, 0.9), chunk(2, "b" * 40, 0.8)]
    kept, total = fit_to_budget(chunks, max_tokens=10_000, model=None)
    assert len(kept) == 2
    assert total == 20
