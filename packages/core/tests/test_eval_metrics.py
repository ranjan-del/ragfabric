"""Retrieval metrics and citation correctness, on hand computed cases."""

import pytest

from ragfabric_core.evaluation.dataset import ExpectedSource
from ragfabric_core.evaluation.metrics import citation_correct, is_relevant, retrieval_scores
from ragfabric_core.evaluation.target import ContextItem


def _ctx(rank: int, document: str, text: str = "filler words here") -> ContextItem:
    return ContextItem(rank=rank, document=document, text=text)


A = ExpectedSource(document="a.md", evidence="ten days of leave")
B = ExpectedSource(document="b.md")


def test_relevant_at_ranks_two_and_four():
    contexts = [
        _ctx(1, "x.md"),
        _ctx(2, "a.md", "Staff get TEN days\n of   leave each year."),
        _ctx(3, "x.md"),
        _ctx(4, "b.md"),
        _ctx(5, "a.md", "unrelated part of a.md"),
    ]
    s = retrieval_scores(contexts, [A, B])
    assert s.precision == pytest.approx(0.4)
    assert s.recall == pytest.approx(1.0)
    assert s.hit is True
    assert s.reciprocal_rank == pytest.approx(0.5)


def test_nothing_relevant_scores_zero_not_none():
    s = retrieval_scores([_ctx(1, "x.md"), _ctx(2, "y.md")], [A])
    assert (s.precision, s.recall, s.hit, s.reciprocal_rank) == (0.0, 0.0, False, 0.0)


def test_nothing_retrieved_scores_zero_with_precision_none():
    s = retrieval_scores([], [A])
    assert (s.precision, s.recall, s.hit, s.reciprocal_rank) == (None, 0.0, False, 0.0)


def test_no_expected_sources_means_unmeasured():
    s = retrieval_scores([_ctx(1, "a.md")], [])
    assert (s.precision, s.recall, s.hit, s.reciprocal_rank) == (None, None, None, None)


def test_half_the_sources_found_is_half_recall():
    s = retrieval_scores([_ctx(1, "b.md")], [A, B])
    assert s.recall == pytest.approx(0.5)
    assert s.reciprocal_rank == pytest.approx(1.0)


def test_right_document_without_the_evidence_is_not_relevant():
    assert is_relevant(_ctx(1, "a.md", "something else"), A) is False
    assert is_relevant(_ctx(1, "a.md", "ten days of leave"), A) is True
    assert is_relevant(_ctx(1, "b.md", "anything"), B) is True


CHUNKS = [
    _ctx(1, "a.md", "Employees receive 10 days of casual leave per year."),
    _ctx(2, "b.md", "The hotel limit is 180 US dollars per night."),
]


def test_a_supported_cited_answer_is_correct():
    check = citation_correct("Employees receive 10 days of casual leave [1].", CHUNKS)
    assert check.correct is True and check.reason is None


def test_a_marker_out_of_range_is_wrong():
    check = citation_correct("Employees receive 10 days of casual leave [3].", CHUNKS)
    assert check.correct is False and "[3]" in check.reason


def test_a_cited_sentence_its_chunk_does_not_support_is_wrong():
    check = citation_correct("Employees receive 10 days of casual leave [2].", CHUNKS)
    assert check.correct is False and "does not support" in check.reason


def test_an_uncited_answer_with_evidence_is_wrong():
    check = citation_correct("Employees receive 10 days of casual leave.", CHUNKS)
    assert check.correct is False and "no [n] marker" in check.reason


def test_a_misquoted_answer_is_wrong():
    check = citation_correct('The policy says "twelve days of casual leave" [1].', CHUNKS)
    assert check.correct is False


def test_no_evidence_answer_is_correct_only_with_nothing_retrieved():
    text = "I could not find this in the documents."
    assert citation_correct(text, []).correct is True
    assert citation_correct(text, CHUNKS).correct is False


def test_an_empty_answer_is_unmeasured():
    assert citation_correct("", CHUNKS).correct is None
