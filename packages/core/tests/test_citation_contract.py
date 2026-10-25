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
        "Staff get 24 days of annual leave [1], and it accrues monthly [2].",
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


def test_a_quote_cited_to_the_wrong_passage_names_the_passage_that_holds_it():
    """The reason is fed back to the model on the retry, so it must say how to repair.

    Phase 11 run 13: "A conscious criminal" was in passage [3] and cited to [1];
    the reason only said the quote was missing, and the retry gave up.
    """
    with pytest.raises(CitationViolation) as exc:
        assert_citation_contract(
            'He is called "a conscious criminal" [1].',
            chunks("Opening statements began.", "The jury was sworn.", "A conscious criminal..."),
        )
    assert "does not appear in any chunk the answer cites" in exc.value.reason
    assert "it appears in passage [3]" in exc.value.reason


def test_a_quote_in_no_passage_at_all_is_not_given_a_passage():
    with pytest.raises(CitationViolation) as exc:
        assert_citation_contract(
            'It says "an invented sentence here" [1].', chunks("Something else.", "Other text.")
        )
    assert "appears in passage" not in exc.value.reason


# A reply that is only markers, "[1]", passed the contract: it cites a passage that
# exists and quotes nothing, yet it says nothing. llama3.2:3b gave exactly that for the
# quickstart's own sample question, and the user saw "[1]" as the answer.


@pytest.mark.parametrize("answer", ["[1]", " [1] [2]. ", "[1]\n"])
def test_an_answer_of_markers_alone_is_a_violation(answer):
    from ragfabric_core.generate.contract import assert_answer_has_words

    with pytest.raises(CitationViolation) as exc:
        assert_answer_has_words(answer)
    assert "only citation markers" in exc.value.reason


@pytest.mark.parametrize(
    "answer",
    [
        "Staff get 24 days [1].",
        "[1] Up to 10 days carry forward.",
        "I could not find an answer to that in the documents provided.",
    ],
)
def test_an_answer_with_words_passes_the_words_check(answer):
    from ragfabric_core.generate.contract import assert_answer_has_words

    assert_answer_has_words(answer)
