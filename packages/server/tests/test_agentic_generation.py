"""Generation over the agent's evidence pool: dropped claims and dated sources.

Two behaviours are under test, and they are deliberately the only two.

**An unsupported claim is removed, not retried.** The Phase 3 citation contract
is reused exactly as it is, applied claim by claim instead of once over the
whole answer. A claim that fails it is deleted and the deletion is recorded.
Nothing routes back into retrieval, because a claim the evidence does not
support is usually not a retrieval failure: it is the model saying more than it
was given, and more retrieval does not make it true.

**Dates are metadata, not a judgement.** Where the chunks answering one
sub-question come from documents with different effective dates, both are
surfaced with their dates. No claim is made about which is correct or whether
they disagree in meaning. That inference is a semantic one, it produces
confident false positives on a small model, and it is deferred to Phase 8 where
it can be measured.
"""

from __future__ import annotations

from ragfabric_core.generate.cited import (
    NO_EVIDENCE_ANSWER,
    SubQuestionEvidence,
    generate_agentic_answer,
    split_claims,
)
from ragfabric_core.generate.contract import assert_citation_contract
from ragfabric_core.providers.offline import ScriptedLLMProvider
from ragfabric_core.strategies.base import RetrievedChunk


def chunk(
    chunk_id: int,
    text: str,
    *,
    document_id: int = 1,
    effective_date: str | None = None,
) -> RetrievedChunk:
    metadata: dict[str, str | int | float | bool | None] = {}
    if effective_date is not None:
        metadata["effective_date"] = effective_date
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        collection_id=None,
        text=text,
        metadata=metadata,
    )


POOL = [
    chunk(1, "The retry limit is five attempts."),
    chunk(2, "A failed batch is escalated to the owner.", document_id=2),
]


def test_an_unsupported_claim_is_removed_not_retried() -> None:
    """The claim goes, the run does not: exactly one model call is made."""
    llm = ScriptedLLMProvider(
        ["The retry limit is five attempts [1]. The board signed this policy in March."]
    )

    answer = generate_agentic_answer("what is the retry limit", POOL, llm)

    assert answer.text == "The retry limit is five attempts [1]."
    assert [claim.text for claim in answer.dropped_claims] == [
        "The board signed this policy in March."
    ]
    assert "marker" in answer.dropped_claims[0].reason
    assert llm.calls == 1


def test_a_fabricated_quote_is_dropped_with_the_reason_recorded() -> None:
    llm = ScriptedLLMProvider(
        ['The retry limit is five attempts [1]. The policy says "nine attempts are allowed" [1].']
    )

    answer = generate_agentic_answer("what is the retry limit", POOL, llm)

    assert answer.text == "The retry limit is five attempts [1]."
    assert len(answer.dropped_claims) == 1
    assert "nine attempts are allowed" in answer.dropped_claims[0].reason


def test_a_claim_citing_a_chunk_that_was_never_retrieved_is_dropped() -> None:
    llm = ScriptedLLMProvider(["The retry limit is five attempts [1]. It resets nightly [9]."])

    answer = generate_agentic_answer("what is the retry limit", POOL, llm)

    assert answer.text == "The retry limit is five attempts [1]."
    assert "[9]" in answer.dropped_claims[0].text


def test_a_fully_supported_answer_is_returned_untouched() -> None:
    llm = ScriptedLLMProvider(
        ["The retry limit is five attempts [1]. Failures escalate to the owner [2]."]
    )

    answer = generate_agentic_answer("what is the retry limit", POOL, llm)

    assert (
        answer.text == "The retry limit is five attempts [1]. Failures escalate to the owner [2]."
    )
    assert answer.dropped_claims == []
    assert answer.generator == "llm"


def test_the_surviving_answer_satisfies_the_citation_contract() -> None:
    """The invariant behind dropping: what is left passes the contract unchanged."""
    llm = ScriptedLLMProvider(
        [
            "The retry limit is five attempts [1]. Nobody reviews it. "
            'It says "nothing at all about reviews" [2].'
        ]
    )

    answer = generate_agentic_answer("what is the retry limit", POOL, llm)

    assert_citation_contract(answer.text, POOL)
    assert len(answer.dropped_claims) == 2


def test_an_answer_with_every_claim_dropped_says_so_rather_than_returning_nothing() -> None:
    llm = ScriptedLLMProvider(["The board signed this policy in March. It resets nightly."])

    answer = generate_agentic_answer("what is the retry limit", POOL, llm)

    assert answer.text == NO_EVIDENCE_ANSWER
    assert len(answer.dropped_claims) == 2


def test_an_explicit_no_evidence_answer_is_not_dropped_for_being_uncited() -> None:
    llm = ScriptedLLMProvider([NO_EVIDENCE_ANSWER])

    answer = generate_agentic_answer("what is the retry limit", POOL, llm)

    assert answer.text == NO_EVIDENCE_ANSWER
    assert answer.dropped_claims == []


def test_an_empty_evidence_pool_makes_no_model_call() -> None:
    llm = ScriptedLLMProvider([])

    answer = generate_agentic_answer("what is the retry limit", [], llm)

    assert answer.text == NO_EVIDENCE_ANSWER
    assert answer.generator == "extractive"
    assert answer.input_tokens == 0
    assert answer.output_tokens == 0
    assert llm.calls == 0


def test_a_quote_containing_a_full_stop_is_not_split_into_two_claims() -> None:
    """The splitter has to respect quotation, or it manufactures uncited fragments."""
    claims = split_claims('The policy says "One. Two." [1]. It also applies here [2].')

    assert [claim.strip() for claim in claims] == [
        'The policy says "One. Two." [1].',
        "It also applies here [2].",
    ]


def test_a_trailing_marker_stays_with_the_claim_it_cites() -> None:
    claims = split_claims("The retry limit is five attempts. [1]")

    assert [claim.strip() for claim in claims] == ["The retry limit is five attempts. [1]"]


DATED_POOL = [
    chunk(1, "The retry limit is three attempts.", document_id=3, effective_date="2024-01-01"),
    chunk(2, "The retry limit is five attempts.", document_id=7, effective_date="2025-06-01"),
]


def test_two_dated_sources_for_one_sub_question_are_both_surfaced() -> None:
    llm = ScriptedLLMProvider(["The retry limit is five attempts [2]."])

    answer = generate_agentic_answer(
        "what is the retry limit",
        DATED_POOL,
        llm,
        sub_question_evidence=[
            SubQuestionEvidence(text="what is the retry limit", chunks=DATED_POOL)
        ],
    )

    assert len(answer.dated_sources) == 1
    reported = answer.dated_sources[0]
    assert reported.sub_question == "what is the retry limit"
    assert [(s.document_id, s.effective_date, s.marker) for s in reported.sources] == [
        (3, "2024-01-01", 1),
        (7, "2025-06-01", 2),
    ]
    assert "2024-01-01" in answer.text
    assert "2025-06-01" in answer.text
    assert_citation_contract(answer.text, DATED_POOL)


def test_one_effective_date_across_the_sources_is_not_surfaced_as_a_difference() -> None:
    same = [
        chunk(1, "The retry limit is five attempts.", document_id=3, effective_date="2024-01-01"),
        chunk(2, "Failures escalate.", document_id=7, effective_date="2024-01-01"),
    ]
    llm = ScriptedLLMProvider(["The retry limit is five attempts [1]."])

    answer = generate_agentic_answer(
        "what is the retry limit",
        same,
        llm,
        sub_question_evidence=[SubQuestionEvidence(text="what is the retry limit", chunks=same)],
    )

    assert answer.dated_sources == []
    assert "2024-01-01" not in answer.text


def test_undated_chunks_produce_no_claim_about_dates() -> None:
    """No date in the metadata means nothing is known, never a date inferred."""
    llm = ScriptedLLMProvider(["The retry limit is five attempts [1]."])

    answer = generate_agentic_answer(
        "what is the retry limit",
        POOL,
        llm,
        sub_question_evidence=[SubQuestionEvidence(text="what is the retry limit", chunks=POOL)],
    )

    assert answer.dated_sources == []


def test_dates_are_reported_per_sub_question_not_across_the_whole_pool() -> None:
    """Two dates under different sub-questions are two separate facts, not one."""
    llm = ScriptedLLMProvider(["The retry limit is three attempts [1]. Failures escalate [2]."])
    pool = [
        chunk(1, "The retry limit is three attempts.", document_id=3, effective_date="2024-01-01"),
        chunk(2, "Failures escalate.", document_id=7, effective_date="2025-06-01"),
    ]

    answer = generate_agentic_answer(
        "what is the retry limit and what happens on failure",
        pool,
        llm,
        sub_question_evidence=[
            SubQuestionEvidence(text="what is the retry limit", chunks=[pool[0]]),
            SubQuestionEvidence(text="what happens on failure", chunks=[pool[1]]),
        ],
    )

    assert answer.dated_sources == []
