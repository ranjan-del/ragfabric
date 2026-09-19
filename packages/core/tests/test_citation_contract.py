import pytest

from ragfabric_core.generate.contract import (
    CitationViolation,
    assert_citation_contract,
    cited_markers,
)
from ragfabric_core.strategies.base import RetrievedChunk


def chunks(*texts: str) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(chunk_id=i, document_id=1, collection_id=None, text=t, score=0.9)
        for i, t in enumerate(texts, start=1)
    ]


def test_cited_markers_reads_every_marker_in_order_without_duplicates():
    assert cited_markers("First [1] then [2] and again [1].") == [1, 2]


def test_a_valid_answer_passes():
    assert_citation_contract(
        'Staff get 24 days of annual leave [1], and it accrues monthly [2].',
        chunks("Employees receive 24 days of annual leave.", "Leave accrues monthly."),
    )


def test_a_marker_pointing_past_the_retrieved_set_is_a_violation():
    with pytest.raises(CitationViolation) as exc:
        assert_citation_contract("The policy says so [4].", chunks("a", "b"))
    assert "marker" in exc.value.reason


def test_a_quote_that_is_not_in_any_cited_chunk_is_a_violation():
    with pytest.raises(CitationViolation) as exc:
        assert_citation_contract(
            'The handbook says "you get forty days of leave" [1].',
            chunks("Employees receive 24 days of annual leave."),
        )
    assert "quote" in exc.value.reason


def test_a_quote_matching_the_chunk_apart_from_whitespace_passes():
    assert_citation_contract(
        'It says "24 days of annual leave" [1].',
        chunks("Employees receive 24   days of\nannual leave."),
    )


def test_a_paraphrase_without_quotation_marks_passes():
    assert_citation_contract(
        "Staff are entitled to just under five weeks off each year [1].",
        chunks("Employees receive 24 days of annual leave."),
    )


def test_an_answer_with_evidence_but_no_marker_is_a_violation():
    with pytest.raises(CitationViolation) as exc:
        assert_citation_contract("Staff get 24 days.", chunks("Employees receive 24 days."))
    assert "uncited" in exc.value.reason


def test_the_no_evidence_sentence_is_allowed_without_a_marker():
    assert_citation_contract(
        "I could not find an answer to that in the documents provided.", chunks("unrelated text")
    )


def test_an_empty_chunk_list_allows_an_uncited_answer():
    assert_citation_contract("I could not find an answer to that.", [])


# --- Quote fidelity floor: short quotes still get checked when they carry a digit ---
#
# The plain 8-character floor let a short, fabricated, digit-bearing quote such as
# "99 days" dodge verification entirely, which is the most dangerous shape of miss:
# short quotes are disproportionately the factual ones (numbers, dates, amounts)
# that a reader trusts a citation for.


def test_a_false_short_numeric_quote_is_a_violation():
    with pytest.raises(CitationViolation) as exc:
        assert_citation_contract(
            'The policy gives "99 days" [1].',
            chunks("Employees receive 24 days of annual leave."),
        )
    assert "quote" in exc.value.reason


def test_a_true_short_numeric_quote_passes():
    assert_citation_contract(
        'The policy gives "24 days" [1].',
        chunks("Employees receive 24 days of annual leave."),
    )


def test_a_short_non_numeric_quote_is_still_unchecked():
    # Documented gap: below the 8-character floor, only digit-bearing quotes are
    # verified. "leave" is not a substring of the chunk below (it says "time
    # off"), so a checked quote would fail here, but this one is short and has
    # no digit, so it is skipped and the answer still passes.
    assert_citation_contract(
        'The handbook calls it "leave" [1].',
        chunks("Employees receive 24 days of annual time off."),
    )


def test_a_long_false_quote_is_still_a_violation():
    with pytest.raises(CitationViolation) as exc:
        assert_citation_contract(
            'The handbook says "you get forty days of leave" [1].',
            chunks("Employees receive 24 days of annual leave."),
        )
    assert "quote" in exc.value.reason
