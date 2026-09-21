import os
from unittest.mock import MagicMock, patch

import pytest

from ragfabric_core.strategies.base import RetrievedChunk
from ragfabric_core.tokens import count_tokens, fit_to_budget, token_count_method

TIKTOKEN_AVAILABLE = os.environ.get("RAGFABRIC_TEST_TIKTOKEN", "")


def chunk(cid: int, text: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=cid, document_id=1, collection_id=None, text=text, score=score)


# Hermetic unit tests (no network calls)


def test_count_tokens_uses_exact_path_when_encoding_exists():
    """Monkeypatched encoding proves exact path without touching tiktoken."""
    stub_encoding = MagicMock()
    stub_encoding.encode.return_value = [1, 2]
    with patch("ragfabric_core.tokens._encoding", return_value=stub_encoding):
        assert count_tokens("hello world", "gpt-4") == 2
        stub_encoding.encode.assert_called_once_with("hello world")


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


def test_token_count_method_returns_exact_when_encoding_exists():
    """Monkeypatched encoding proves method selection without touching tiktoken."""
    stub_encoding = MagicMock()
    with patch("ragfabric_core.tokens._encoding", return_value=stub_encoding):
        assert token_count_method("gpt-4") == "exact"


def test_token_count_method_returns_estimate_for_unknown_model():
    assert token_count_method("llama3.2:3b") == "estimate"


def test_token_count_method_returns_estimate_for_none():
    assert token_count_method(None) == "estimate"


def test_count_tokens_falls_back_to_estimate_when_download_fails():
    """Air-gapped case: encoding exists but cannot be fetched, falls back to estimate.

    Simulates a network failure (ConnectionError) that would occur when trying
    to download the encoding in an offline or air-gapped deployment.
    """
    try:
        import requests
    except ImportError:
        pytest.skip("requests not available (should not occur)")

    with patch("ragfabric_core.tokens._encoding") as mock_encoding:
        mock_encoding.side_effect = requests.exceptions.ConnectionError("Network unreachable")
        # 11 characters // 4 = 2 tokens (estimate)
        assert count_tokens("hello world", "text-embedding-3-small") == 2


def test_token_count_method_returns_estimate_when_download_fails():
    """Air-gapped case: encoding download fails, method correctly reports estimate.

    This proves token_count_method and count_tokens are in sync: both degrade
    to estimate when the download fails, rather than raising.
    """
    try:
        import requests
    except ImportError:
        pytest.skip("requests not available (should not occur)")

    with patch("ragfabric_core.tokens._encoding") as mock_encoding:
        mock_encoding.side_effect = requests.exceptions.ConnectionError("Network unreachable")
        assert token_count_method("text-embedding-3-small") == "estimate"


# Integration tests (require RAGFABRIC_TEST_TIKTOKEN and may touch network)

pytestmark_integration = [
    pytest.mark.integration,
    pytest.mark.skipif(not TIKTOKEN_AVAILABLE, reason="needs RAGFABRIC_TEST_TIKTOKEN"),
]


@pytest.mark.integration
@pytest.mark.skipif(not TIKTOKEN_AVAILABLE, reason="needs RAGFABRIC_TEST_TIKTOKEN")
def test_count_tokens_is_exact_for_text_embedding_3_small():
    assert count_tokens("hello world", "text-embedding-3-small") == 2
