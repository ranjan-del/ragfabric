"""A number, code or quote cited to the wrong passage (found in eval answers, 22 October 2026).

Real cases, over passages from the shipped corpus:

* llama3.1:8b, q-020: "claims submitted later than 30 days [4]" where passage 4 is the IT runbook and
  the 30 days rule is passage 3.
* llama3.2:3b, q-020: the same fact cited as "passage [4]".
* llama3.2:3b, q-009: "the product's code is BWA-410" cited to the project note [2]; the code is in
  the product catalogue, passage [4], and the delay claim in the same sentence is rightly [2].

The contract checked quotes against the joined cited passages only, so none of these was caught.
"""

import pytest

from ragfabric_core.evaluation.metrics import citation_correct
from ragfabric_core.evaluation.target import ContextItem
from ragfabric_core.generate.attribution import (
    assert_number_attribution,
    find_misattributions,
    repair_attribution,
)
from ragfabric_core.generate.cited import generate_cited_answer
from ragfabric_core.generate.contract import CitationViolation, assert_citation_contract
from ragfabric_core.providers.base import Completion
from ragfabric_core.strategies.base import RetrievedChunk

LEAVE_APPROVAL = (
    "year. A medical certificate is required for sick leave longer than 2 consecutive days.\n\n"
    "## Approval\n\nLeave requests are approved by the employee's direct manager. Requests longer "
    "than 5 working days also need approval from the People Operations team."
)
LEAVE_2026_TAIL = (
    "eave per year. A medical certificate is required for sick leave longer than 3 consecutive "
    "days.\n\n## Parental leave\n\nNew in 2026: parents receive 16 weeks of paid parental leave.\n\n"
    "## Approval\n\nLeave requests are approved by the employee's direct manager. Requests longer "
    "than 5 working days also need approval from the People Operations team."
)
TRAVEL = (
    "# Travel and Expenses Policy\n\n## Limits\n\nThe daily meal limit is 60 US dollars when "
    "travelling within the country and 90 US dollars when travelling abroad.\nThe hotel limit is "
    "180 US dollars per night.\n\n## Approval\n\nAny trip longer than 5 days needs approval from "
    "the employee's director in addition to their direct manager.\n\n## Claims\n\nExpense claims "
    "must be submitted within 30 days of the end of the trip. Claims submitted later are paid only "
    "with approval from the Chief Financial Officer."
)
RUNBOOK_TAIL = (
    "er's laptop failed disk encryption checks. Reimage the laptop within 2 working days.\n\n"
    "BW-5120: The VPN rejected a certificate that has expired. Certificates are valid for 365 "
    "days.\n\n## On-call\n\nThe IT Operations on-call rotation changes every Monday at 09:00. Ivan "
    "Petrov maintains the rota."
)
BEACON = (
    "# Project Beacon\n\nProject Beacon builds the demand forecasting service sold to customers as "
    "the Forecast Pro product.\n\n## Dependencies\n\nProject Beacon cannot launch until Project "
    "Atlas is complete, because the forecasting models train on the new columnar warehouse."
)
ATLAS = (
    "# Project Atlas\n\nProject Atlas is owned by the Data Platform team.\n\n## Budget and "
    "schedule\n\nThe approved budget for Project Atlas is 420,000 US dollars. The migration is due "
    "to finish in September 2026.\n\n## Risks\n\nThe main risk is double running cost, estimated "
    "at 18,000 US dollars per month."
)
CATALOGUE = (
    "# Product Catalogue 2026\n\n| Code | Product | Price |\n|---|---|---|\n| BWA-200 | Insight "
    "Team | 199 |\n| BWA-410 | Forecast Pro, demand forecasting add-on | 350 |\n\nCustomers who pay "
    "annually in advance receive a 15 percent discount on the monthly price."
)


def chunks(*texts: str) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(chunk_id=i, document_id=1, collection_id=None, text=t, score=0.9)
        for i, t in enumerate(texts, start=1)
    ]


APPROVAL_SET = chunks(LEAVE_APPROVAL, LEAVE_2026_TAIL, TRAVEL, RUNBOOK_TAIL, CATALOGUE)
# Case 1, and case 2 with the 3b model's wording: the 30 days rule is in passage 3, cited as [4].
Q020_8B = (
    "Requests longer than 5 working days [1], [2], any trip longer than 5 days [3], and claims "
    "submitted later than 30 days [4]."
)
Q020_3B = (
    "According to passage [1], leave requests longer than 5 working days need People Operations "
    "approval. According to passage [4], expense claims submitted later than 30 days of the end of "
    "the trip are paid only with approval from the Chief Financial Officer."
)
PRODUCT_SET = chunks(ATLAS, BEACON, RUNBOOK_TAIL, CATALOGUE, TRAVEL)
Q009_3B = "Project Beacon is delayed if Project Atlas slips, and the product's code is BWA-410 [2]."


def test_a_number_cited_to_a_passage_without_it_is_found():
    found = find_misattributions(Q020_8B, APPROVAL_SET)
    assert [(m.phrase, m.cited, m.holders) for m in found] == [("30 days", (4,), (3,))]


def test_the_number_is_moved_to_the_passage_that_holds_it():
    repaired = repair_attribution(Q020_8B, APPROVAL_SET)
    assert repaired.text.endswith("claims submitted later than 30 days [3].")
    assert "[4]" not in repaired.text
    assert "[1], [2]" in repaired.text and "any trip longer than 5 days [3]" in repaired.text
    assert repaired.unresolved == [] and len(repaired.repairs) == 1
    assert find_misattributions(repaired.text, APPROVAL_SET) == []


def test_the_wording_of_the_3b_answer_is_repaired_too():
    repaired = repair_attribution(Q020_3B, APPROVAL_SET)
    assert "According to passage [3], expense claims submitted later than 30 days" in repaired.text
    assert "According to passage [1], leave requests longer than 5 working days" in repaired.text
    assert find_misattributions(repaired.text, APPROVAL_SET) == []


def test_a_sentence_with_two_facts_keeps_the_right_marker_and_gains_the_holder():
    repaired = repair_attribution(Q009_3B, PRODUCT_SET)
    assert repaired.text == (
        "Project Beacon is delayed if Project Atlas slips, and the product's code is BWA-410 "
        "[2], [4]."
    )


def test_a_number_that_is_in_the_cited_passage_is_left_alone():
    answer = "The hotel limit is 180 US dollars per night [3]."
    assert repair_attribution(answer, APPROVAL_SET).text == answer


def test_a_computed_number_that_no_passage_holds_is_left_alone():
    answer = "Three months of 18,000 US dollars per month is 54,000 US dollars extra [1]."
    out = repair_attribution(answer, PRODUCT_SET)
    assert out.text == answer and out.unresolved == []


def test_a_number_the_question_restates_is_not_a_fact_to_locate():
    # "3 months" is not in any passage, even though "3 consecutive days" is: no unit match.
    answer = "That adds 3 months of double running cost [1]."
    assert repair_attribution(answer, APPROVAL_SET).text == answer


def test_a_year_is_context_not_a_fact():
    answer = "In 2025 a medical certificate was needed after 2 consecutive days [1]."
    assert repair_attribution(answer, APPROVAL_SET).text == answer


def test_a_quote_cited_to_the_wrong_passage_is_moved():
    answer = 'The rule is "paid only with approval from the Chief Financial Officer" [4].'
    repaired = repair_attribution(answer, APPROVAL_SET)
    assert repaired.text.endswith('Financial Officer" [3].')


def test_a_fact_held_by_two_passages_equally_is_unresolved_not_guessed():
    two = chunks("The notice period is 30 days.", "The refund window is 30 days.", "Unrelated.")
    answer = "It is 30 days [3]."
    out = repair_attribution(answer, two)
    assert out.text == answer
    assert [m.phrase for m in out.unresolved] == ["30 days"]
    with pytest.raises(CitationViolation) as exc:
        assert_number_attribution(answer, two)
    assert "30 days" in exc.value.reason and "passage [1]" in exc.value.reason


def test_the_overlap_with_the_sentence_picks_between_two_holders():
    two = chunks(
        "The notice period for resignation is 30 days.", "The refund window is 30 days.", "x"
    )
    out = repair_attribution("The refund window is 30 days [3].", two)
    assert out.text == "The refund window is 30 days [2]."


class ScriptedLLM:
    name = "scripted"
    default_model = "scripted"

    def __init__(self, *texts: str) -> None:
        self._texts = list(texts)
        self.calls = 0

    def complete(self, messages, *, model=None, max_tokens=1024, temperature=0.0, json_schema=None):
        self.calls += 1
        return Completion(
            text=self._texts.pop(0),
            model="scripted",
            provider="scripted",
            input_tokens=10,
            output_tokens=5,
            latency_ms=1,
        )


def test_generation_repairs_the_marker_without_a_second_call():
    llm = ScriptedLLM(Q020_8B)
    out = generate_cited_answer("what needs approval", APPROVAL_SET, llm)
    assert llm.calls == 1 and out.retried is False and out.generator == "llm"
    assert out.text.endswith("30 days [3].")
    assert out.repaired_citations == ["30 days: [4] to [3]"]


def test_generation_repairs_a_quote_the_contract_used_to_retry():
    llm = ScriptedLLM('The rule is "paid only with approval from the Chief Financial Officer" [4].')
    out = generate_cited_answer("claims", APPROVAL_SET, llm)
    assert llm.calls == 1 and out.text.endswith('Officer" [3].')


def test_an_unresolvable_misattribution_goes_to_the_existing_repair_retry():
    two = chunks("The notice period is 30 days.", "The refund window is 30 days.", "Unrelated.")
    llm = ScriptedLLM("It is 30 days [3].", "It is 30 days [1].")
    out = generate_cited_answer("period", two, llm)
    assert llm.calls == 2 and out.retried is True and out.text == "It is 30 days [1]."


def test_the_contract_itself_is_unchanged_for_other_answers():
    assert_citation_contract(Q020_8B, APPROVAL_SET)  # the new rule is separate


def _items(texts):
    return [ContextItem(rank=i, document="d.md", text=t) for i, t in enumerate(texts, start=1)]


def test_the_citation_metric_fails_a_number_cited_to_the_wrong_passage():
    items = _items([c.text for c in APPROVAL_SET])
    wrong = citation_correct(Q020_8B, items)
    assert wrong.correct is False and "30 days" in wrong.reason
    right = citation_correct(repair_attribution(Q020_8B, APPROVAL_SET).text, items)
    assert right.correct is True


def test_the_agent_path_moves_the_marker_instead_of_dropping_nothing():
    from ragfabric_core.generate.cited import apply_agentic_contract

    out = apply_agentic_contract(Q020_8B, APPROVAL_SET)
    assert out.text.endswith("30 days [3].") and out.dropped_claims == []
