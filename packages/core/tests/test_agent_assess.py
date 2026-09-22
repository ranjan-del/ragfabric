"""The assess node, and why it is the node worth being pessimistic in.

Assessment is where an agent decides it is done. A permissive assessor does not
produce a slightly worse answer, it produces a confident wrong one: the loop
stops, the gap is never chased, and nothing downstream knows a gap existed. So
the rubric is strict in the prompt, and the two failures a small model actually
makes are handled in code rather than asked for politely. A verdict that claims
an answer while naming what is missing is not trusted, and a sub-question the
model simply did not mention stays open.
"""

from __future__ import annotations

import json

from agent_doubles import RecordingLLM, chunk

from ragfabric_core.agent.nodes import assess
from ragfabric_core.agent.state import AgentState, NodeName, SubQuestion, SubQuestionStatus


def assess_json(*verdicts: dict) -> str:
    return json.dumps({"verdicts": list(verdicts)})


def state_with(*texts: str) -> AgentState:
    state = AgentState(
        question="the question",
        sub_questions=[SubQuestion(text=text, tool="semantic_search") for text in texts],
    )
    state.add_evidence([chunk(1, text="retries are discussed at length in this policy")])
    return state


def test_evidence_that_answers_a_sub_question_closes_it():
    state = state_with("what is the retry limit")
    llm = RecordingLLM(
        assess_json({"sub_question": "what is the retry limit", "answered": True, "missing": None})
    )

    outcome = assess(state, llm=llm)

    assert state.sub_questions[0].status is SubQuestionStatus.ANSWERED
    assert outcome.verdicts.answered == [0]
    assert state.node_llm_calls[NodeName.ASSESS] == 1


def test_topically_related_evidence_is_not_counted_as_answered():
    """The rubric has to reach the model, and a hedged verdict is read as a no.

    A verdict that says answered while also naming something missing is the
    contradiction small models produce constantly. Reading it the permissive
    way is exactly how an agent ends up confidently wrong, so it is read the
    other way.
    """
    state = state_with("what is the retry limit")
    llm = RecordingLLM(
        assess_json(
            {
                "sub_question": "what is the retry limit",
                "answered": True,
                "missing": "the actual number of retries",
            }
        )
    )

    outcome = assess(state, llm=llm)

    assert state.sub_questions[0].status is SubQuestionStatus.OPEN
    assert outcome.verdicts.answered == []
    assert outcome.verdicts.missing[0] == "the actual number of retries"

    prompt = llm.last_prompt().lower()
    assert "related" in prompt
    assert "unsure" in prompt


def test_a_missing_description_is_carried_into_the_repair():
    state = state_with("what is the retry limit")
    llm = RecordingLLM(
        assess_json(
            {
                "sub_question": "what is the retry limit",
                "answered": False,
                "missing": "the number of retries allowed",
            }
        )
    )

    outcome = assess(state, llm=llm)

    assert outcome.verdicts.missing == {0: "the number of retries allowed"}
    assert state.sub_questions[0].status is SubQuestionStatus.OPEN


def test_a_malformed_assessment_leaves_the_sub_question_open():
    """A model that cannot be read has not said the question is answered."""
    state = state_with("what is the retry limit")
    llm = RecordingLLM("Honestly it is hard to say, the policy is quite long.")

    outcome = assess(state, llm=llm)

    assert outcome.violation is not None
    assert outcome.violation.contract == "assess"
    assert state.sub_questions[0].status is SubQuestionStatus.OPEN
    assert outcome.verdicts.unjudged == [0]
    assert outcome.verdicts.missing[0]


def test_a_sub_question_the_model_ignored_stays_open():
    """Silence is not consent. An unjudged part must not be quietly closed."""
    state = state_with("what is the retry limit", "what is the backoff interval")
    llm = RecordingLLM(
        assess_json({"sub_question": "what is the retry limit", "answered": True, "missing": None})
    )

    outcome = assess(state, llm=llm)

    assert state.sub_questions[0].status is SubQuestionStatus.ANSWERED
    assert state.sub_questions[1].status is SubQuestionStatus.OPEN
    assert outcome.verdicts.unjudged == [1]


def test_a_verdict_for_an_unknown_sub_question_is_ignored():
    """A model that invents a sub-question must not close a real one by accident."""
    state = state_with("what is the retry limit")
    llm = RecordingLLM(
        assess_json({"sub_question": "something else entirely", "answered": True, "missing": None})
    )

    assess(state, llm=llm)

    assert state.sub_questions[0].status is SubQuestionStatus.OPEN


def test_already_resolved_sub_questions_cost_no_call():
    state = state_with("what is the retry limit")
    state.sub_questions[0].mark_answered()
    llm = RecordingLLM()

    outcome = assess(state, llm=llm)

    assert llm.calls == 0
    assert state.llm_calls == 0
    assert outcome.span.attributes["skipped"] is True


def test_the_evidence_pool_is_what_is_judged():
    state = state_with("what is the retry limit")
    state.add_evidence([chunk(2, text="the retry limit is five attempts")])
    llm = RecordingLLM(
        assess_json({"sub_question": "what is the retry limit", "answered": True, "missing": None})
    )

    assess(state, llm=llm)

    prompt = llm.last_prompt()
    assert "the retry limit is five attempts" in prompt
    assert "retries are discussed at length" in prompt


def test_the_strict_rubric_is_what_the_model_is_sent_by_default():
    """The setting is proved by the prompt the provider received, not by a field.

    ``assess_strictness`` was configurable, validated and printed back by
    ``config validate`` while both settings produced the same prompt. A test
    that asserted the value was stored would have passed throughout.
    """
    state = state_with("what is the retry limit")
    llm = RecordingLLM(
        assess_json({"sub_question": "what is the retry limit", "answered": True, "missing": None})
    )

    assess(state, llm=llm)

    prompt = llm.last_prompt()
    assert "The rubric is strict" in prompt
    assert "Related is not answered." in prompt
    assert "Lenient is not credulous." not in prompt


def test_the_lenient_rubric_reaches_the_model_when_it_is_configured():
    state = state_with("what is the retry limit")
    llm = RecordingLLM(
        assess_json({"sub_question": "what is the retry limit", "answered": True, "missing": None})
    )

    assess(state, llm=llm, strictness="lenient")

    prompt = llm.last_prompt()
    assert "The rubric is lenient" in prompt
    assert "Lenient is not credulous." in prompt
    assert "Related is not answered." not in prompt


def test_the_two_rubrics_disagree_about_evidence_that_only_implies_the_answer():
    """Different words are not enough. The instruction itself has to differ.

    Strict tells the model that evidence which implies the answer is not an
    answer. Lenient tells it the opposite. Anything less than that is two
    spellings of one rubric.
    """
    state = state_with("what is the retry limit")
    strict = RecordingLLM(
        assess_json({"sub_question": "what is the retry limit", "answered": False, "missing": "x"})
    )
    lenient = RecordingLLM(
        assess_json({"sub_question": "what is the retry limit", "answered": True, "missing": None})
    )

    assess(state_with("what is the retry limit"), llm=strict, strictness="strict")
    assess(state, llm=lenient, strictness="lenient")

    assert "guess it, is NOT answered." in strict.last_prompt()
    assert "draw it is answered" in lenient.last_prompt()
