import json

from agent_doubles import RecordingLLM

from ragfabric_core.router.classifier import ClassifierOutcome, classify, one_sentence
from ragfabric_core.router.signals import extract_signals
from ragfabric_core.strategies.base import StrategyName as S

ALL = [S.TRADITIONAL, S.VECTORLESS, S.AGENTIC, S.GRAPH]


def reply(**fields):
    body = {
        "query_type": "multi_hop",
        "strategy": "agentic",
        "confidence": 0.7,
        "reasoning": "It needs facts from two documents.",
    } | fields
    return json.dumps(body)


def signals(q="Tell me everything about how onboarding and payroll interact for contractors"):
    return extract_signals(q, relation_types=["OWNS"])


def test_a_valid_reply_is_returned_with_its_real_token_counts():
    llm = RecordingLLM(reply())
    outcome = classify("q", signals(), llm=llm, available=ALL)
    assert outcome.reply.strategy == "agentic" and outcome.llm_calls == 1
    assert outcome.provider == llm.name
    assert outcome.input_tokens > 0 and outcome.output_tokens > 0


def test_the_prompt_carries_the_question_and_signals_and_no_document_text():
    llm = RecordingLLM(reply())
    classify("What is onboarding?", signals(), llm=llm, available=ALL)
    prompt = "\n".join(m.content for m in llm.prompts[0])
    assert "What is onboarding?" in prompt and "chunk" not in prompt.lower()


def test_malformed_json_is_a_violation_not_an_exception():
    outcome = classify("q", signals(), llm=RecordingLLM("I think agentic"), available=ALL)
    assert outcome.reply is None and outcome.violation is not None and outcome.llm_calls == 1


def test_a_strategy_this_request_cannot_serve_is_a_violation():
    outcome = classify(
        "q",
        signals(),
        llm=RecordingLLM(reply(strategy="graph")),
        available=[S.TRADITIONAL, S.VECTORLESS, S.AGENTIC],
    )
    assert outcome.reply is None and "graph" in outcome.violation.error


def test_a_provider_error_is_reported_not_raised():
    class Broken(RecordingLLM):
        def complete(self, *args, **kwargs):
            raise TimeoutError("model did not answer")

    outcome = classify("q", signals(), llm=Broken(), available=ALL)
    assert outcome.error and outcome.reply is None and outcome.llm_calls == 1


def test_one_sentence_keeps_the_first_sentence_and_the_length_cap():
    assert one_sentence("First part. Second part.") == "First part."
    assert len(one_sentence("x" * 500)) <= 200


def test_a_multi_sentence_reasoning_is_trimmed_to_one_sentence():
    llm = RecordingLLM(reply(reasoning="Two documents are needed. Also a second thought."))
    outcome = classify("q", signals(), llm=llm, available=ALL)
    assert outcome.reply.reasoning == "Two documents are needed."


def test_an_empty_outcome_is_valid():
    assert ClassifierOutcome().llm_calls == 0


def test_a_reasoning_that_is_blank_is_a_violation_not_a_reply():
    outcome = classify("q", signals(), llm=RecordingLLM(reply(reasoning="   ")), available=ALL)
    assert outcome.reply is None and outcome.llm_calls == 1
    assert outcome.violation is not None and "reasoning was empty" in outcome.violation.error
