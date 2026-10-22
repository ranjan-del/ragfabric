"""The two judges: a fixed-rubric LLM judge and a deterministic lexical one."""

import json

import pytest

from ragfabric_core.evaluation.judge import LexicalJudge, LLMJudge, build_judge
from ragfabric_core.evaluation.target import ContextItem
from ragfabric_core.providers.offline import OfflineLLMProvider, ScriptedLLMProvider

CTX = [
    ContextItem(rank=1, document="a.md", text="Employees receive 10 days of casual leave."),
    ContextItem(rank=2, document="b.md", text="The hotel limit is 180 dollars per night."),
]


class Recording(ScriptedLLMProvider):
    def __init__(self, responses):
        super().__init__(responses=responses, model="judge-model")
        self.temperatures = []
        self.models = []
        self.schemas = []
        self.prompts = []

    def complete(self, messages, *, model=None, max_tokens=1024, temperature=0.0, json_schema=None):
        self.temperatures.append(temperature)
        self.models.append(model)
        self.schemas.append(json_schema)
        self.prompts.append(messages[-1].content)
        return super().complete(
            messages, model=model, max_tokens=max_tokens, temperature=temperature
        )


def _reply(score, reason="because"):
    return json.dumps({"score": score, "reason": reason})


def _claims(*verdicts, reason=None):
    body = {"claims": [{"claim": f"claim {i}", "supported": v} for i, v in enumerate(verdicts)]}
    return json.dumps(body)


def test_the_llm_judge_scores_each_rubric_once():
    llm = Recording([_reply(0.8), _claims(True), _reply(0.5)])
    judge = LLMJudge(llm, model="judge-model")
    s = judge.score("How many days?", "10 days.", "Ten days [1].", CTX)
    assert (s.correctness, s.faithfulness, s.context_relevance) == (0.8, 1.0, 0.5)
    assert s.reasons["correctness"] == "because"
    assert llm.calls == 3 and s.calls == 3
    assert llm.temperatures == [0.0, 0.0, 0.0]
    assert llm.models == ["judge-model"] * 3


def test_a_reply_that_is_not_json_is_unmeasured():
    llm = Recording(["not json", "```json\n" + _claims(True) + "\n```", _reply(0.5)])
    s = LLMJudge(llm, model=None).score("q", "e", "a [1].", CTX)
    assert s.correctness is None and "not JSON" in s.reasons["correctness"]
    assert s.faithfulness == 1.0


def test_a_score_out_of_range_is_unmeasured():
    llm = Recording([_reply(1.4), _claims(), _reply("high")])
    s = LLMJudge(llm, model=None).score("q", "e", "a [1].", CTX)
    assert (s.correctness, s.faithfulness, s.context_relevance) == (None, None, None)


def test_no_contexts_skips_the_context_rubrics():
    llm = Recording([_reply(0.0)])
    s = LLMJudge(llm, model=None).score("q", "e", "I could not find this.", [])
    assert s.correctness == 0.0
    assert s.faithfulness is None and s.context_relevance is None
    assert llm.calls == 1


def test_lexical_correctness_is_token_f1():
    # answer words {employees, receive, 10, days}, expected {10, days, casual, leave}
    s = LexicalJudge().score("q", "10 days casual leave", "Employees receive 10 days [1].", CTX)
    # overlap 2, precision 2/4, recall 2/4, f1 0.5
    assert s.correctness == pytest.approx(0.5)


def test_lexical_faithfulness_is_the_share_of_supported_sentences():
    answer = "Employees receive 10 days of casual leave [1]. Parrots fly south quickly [1]."
    s = LexicalJudge().score("q", "10 days", answer, CTX)
    assert s.faithfulness == pytest.approx(0.5)


def test_lexical_context_relevance_is_the_share_of_contexts_sharing_an_answer_word():
    s = LexicalJudge().score("q", "10 days of casual leave", "x [1].", CTX)
    assert s.context_relevance == pytest.approx(0.5)


def test_lexical_scores_are_unmeasured_without_inputs():
    s = LexicalJudge().score("q", "", "I could not find this.", [])
    assert (s.correctness, s.faithfulness, s.context_relevance) == (None, None, None)


def test_auto_picks_lexical_offline_and_llm_otherwise():
    assert build_judge("auto", OfflineLLMProvider(), None).kind == "lexical"
    assert build_judge("auto", ScriptedLLMProvider([]), "m").kind == "llm"
    assert build_judge("lexical", ScriptedLLMProvider([]), None).kind == "lexical"
    judge = build_judge("llm", ScriptedLLMProvider([]), "judge-x")
    assert judge.kind == "llm" and judge.model == "judge-x"


def test_an_llm_judge_cannot_run_offline():
    with pytest.raises(ValueError, match="offline"):
        build_judge("llm", OfflineLLMProvider(), None)


# Faithfulness is the share of claims the judge marks supported, computed here. The v1 judge
# emitted the number itself, before its reason, and gave 0 to answers that were supported
# (measured: 10 of 73 hand-labelled answers; see tests/fixtures/faithfulness_labels.json).


def _faithfulness(reply):
    llm = Recording([reply])
    judge = LLMJudge(llm, model=None, metrics=("faithfulness",))
    s = judge.score("q", "e", "Ten days [1].", CTX)
    return s, llm


def test_faithfulness_is_the_share_of_supported_claims():
    s, llm = _faithfulness(_claims(True, True, False, True))
    assert s.faithfulness == 0.75
    assert llm.calls == 1


def test_every_claim_supported_is_one_never_zero():
    s, _ = _faithfulness(_claims(True))
    assert s.faithfulness == 1.0


def test_no_claim_supported_is_zero_only_when_the_judge_said_so():
    s, _ = _faithfulness(_claims(False, False))
    assert s.faithfulness == 0.0


def test_the_faithfulness_request_asks_for_verdicts_not_a_number():
    _, llm = _faithfulness(_claims(True))
    schema = llm.schemas[0]
    assert "score" not in schema["properties"]
    item = schema["properties"]["claims"]["items"]
    assert list(item["properties"]) == ["claim", "supported"]
    assert item["properties"]["supported"] == {"type": "boolean"}


@pytest.mark.parametrize(
    "reply",
    [
        json.dumps({"claims": []}),
        json.dumps({"claims": [{"claim": "a", "supported": "yes"}]}),
        json.dumps({"claims": [{"claim": "a"}]}),
        json.dumps({"claims": "none"}),
        json.dumps({"score": 0, "reason": "supported"}),
        "not json",
    ],
)
def test_a_faithfulness_reply_that_is_not_verdicts_is_unmeasured(reply):
    s, _ = _faithfulness(reply)
    assert s.faithfulness is None
    assert s.reasons["faithfulness"]


def test_the_unsupported_claims_are_the_stored_reason():
    s, _ = _faithfulness(_claims(True, False))
    assert "claim 1" in s.reasons["faithfulness"]
    assert "claim 0" not in s.reasons["faithfulness"]


def test_citation_markers_and_refusals_are_not_judged_as_claims():
    llm = Recording([])
    judge = LLMJudge(llm, model=None, metrics=("faithfulness",))
    s = judge.score("q", "e", "I could not find an answer to that in the documents provided.", CTX)
    assert s.faithfulness is None and llm.calls == 0
