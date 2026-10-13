from ragfabric_core.generate.cited import generate_cited_answer
from ragfabric_core.providers.base import Completion
from ragfabric_core.strategies.base import RetrievedChunk


def chunks(*texts: str) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(chunk_id=i, document_id=1, collection_id=None, text=t, score=0.9)
        for i, t in enumerate(texts, start=1)
    ]


class ScriptedLLM:
    name = "scripted"
    default_model = "scripted"

    def __init__(self, *texts: str) -> None:
        self._texts = list(texts)
        self.prompts: list[str] = []
        self.conversations: list[list] = []

    def complete(self, messages, *, model=None, max_tokens=1024, temperature=0.0, json_schema=None):
        self.prompts.append(messages[-1].content)
        self.conversations.append(list(messages))
        text = self._texts.pop(0) if self._texts else ""
        return Completion(
            text=text,
            model="scripted",
            provider="scripted",
            input_tokens=10,
            output_tokens=5,
            latency_ms=3,
        )


def test_a_valid_first_answer_is_returned_as_is():
    llm = ScriptedLLM("Staff get 24 days of annual leave [1].")
    out = generate_cited_answer(
        "how much leave", chunks("Employees receive 24 days of annual leave."), llm
    )
    assert out.text == "Staff get 24 days of annual leave [1]."
    assert out.generator == "llm"
    assert out.retried is False
    # Ruling A: exactly one call was made, so the counts are exactly its counts.
    assert out.input_tokens == 10 and out.output_tokens == 5


def test_the_numbered_passages_reach_the_prompt():
    llm = ScriptedLLM("Answer [1].")
    generate_cited_answer("q", chunks("first passage", "second passage"), llm)
    prompt = llm.prompts[0]
    assert "[1] first passage" in prompt
    assert "[2] second passage" in prompt


def test_a_contract_violation_is_retried_once_with_the_reason_in_the_prompt():
    llm = ScriptedLLM("The policy says so [9].", "Staff get 24 days [1].")
    out = generate_cited_answer("q", chunks("Employees receive 24 days."), llm)
    assert out.text == "Staff get 24 days [1]."
    assert out.retried is True
    assert "marker" in llm.prompts[1]
    # Ruling A: two calls were actually made (one rejected, one accepted), so
    # both must be counted, the generator is the LLM path (not a fallback),
    # and retried must be true because a second call was really issued.
    assert out.generator == "llm"
    assert out.input_tokens == 20 and out.output_tokens == 10


def test_the_retry_shows_the_model_the_answer_it_must_repair():
    """A named failure is only repairable if the model can see what it wrote.

    Phase 11 run 13 (MultiHop-RAG, llama3.2:3b): the retry carried the reason
    but not the rejected answer, so the model regenerated from scratch and
    took the no-evidence exit, losing answers that were right but cited badly.
    """
    rejected = "Staff get 24 days of leave."
    llm = ScriptedLLM(rejected, "Staff get 24 days of leave [1].")
    out = generate_cited_answer("q", chunks("Employees receive 24 days."), llm)
    assert out.text == "Staff get 24 days of leave [1]."
    first, second = llm.conversations
    # The original question and passages are sent again unchanged ...
    assert [m.role for m in second[:2]] == ["system", "user"]
    assert second[1].content == first[1].content
    # ... followed by the rejected answer as the model's own turn ...
    assert second[2].role == "assistant" and second[2].content == rejected
    # ... and a correction that names the reason and asks for a repair, not a rewrite.
    assert second[3].role == "user"
    correction = second[3].content
    assert "no [n] marker" in correction
    assert "correct" in correction.lower() and "citation" in correction.lower()
    assert "only if the passages do not answer" in correction.lower()


def test_two_violations_fall_back_to_the_extractive_generator():
    llm = ScriptedLLM("nonsense [9].", "still nonsense [9].")
    out = generate_cited_answer("leave", chunks("Employees receive 24 days of annual leave."), llm)
    assert out.generator == "extractive"
    assert out.retried is True
    assert "[1]" in out.text
    # Ruling A: both rejected LLM calls actually happened and must be
    # counted even though the returned text came from the extractive
    # generator, which makes no LLM call of its own and adds nothing.
    assert out.input_tokens == 20 and out.output_tokens == 10


def test_the_extractive_fallback_can_be_switched_off():
    import pytest

    from ragfabric_core.generate.contract import CitationViolation

    llm = ScriptedLLM("nonsense [9].", "still nonsense [9].")
    with pytest.raises(CitationViolation):
        generate_cited_answer("q", chunks("text"), llm, extractive_fallback=False)


def test_no_chunks_produces_the_no_evidence_sentence_without_calling_the_model():
    llm = ScriptedLLM()
    out = generate_cited_answer("q", [], llm)
    assert "could not find" in out.text.lower()
    assert llm.prompts == []
    assert out.generator == "extractive"
    # Ruling A: no call was made at all on this path, so both counts must be
    # exactly zero and retried must be false, never inferred from the
    # extractive_fallback default.
    assert out.input_tokens == 0 and out.output_tokens == 0
    assert out.retried is False


def _offline_llm():
    from ragfabric_core.config_file import LLMConfig
    from ragfabric_core.providers.registry import build_llm_provider

    return build_llm_provider(LLMConfig(provider="offline"), env={})


def test_the_configured_offline_provider_answers_extractively():
    """``llm.provider: offline`` must answer, not 500: ADR 0004 promises extractive answers."""
    from ragfabric_core.providers.base import is_offline

    llm = _offline_llm()
    assert is_offline(llm)
    out = generate_cited_answer(
        "how much annual leave", chunks("Employees receive 24 days of annual leave."), llm
    )
    assert out.generator == "extractive"
    assert "24 days of annual leave" in out.text and "[1]" in out.text
    assert out.input_tokens == 0 and out.output_tokens == 0
    assert llm.calls == 0


def test_an_exhausted_scripted_double_still_fails_loudly():
    """A test double whose script ran out is not the offline provider: it must still raise."""
    import pytest

    from ragfabric_core.providers.base import ProviderError, is_offline
    from ragfabric_core.providers.offline import ScriptedLLMProvider

    llm = ScriptedLLMProvider(responses=[])
    assert not is_offline(llm)
    with pytest.raises(ProviderError, match="script exhausted"):
        generate_cited_answer("how much leave", chunks("Employees receive 24 days."), llm)


def test_a_scripted_provider_with_responses_still_takes_the_llm_path():
    from ragfabric_core.providers.offline import ScriptedLLMProvider

    llm = ScriptedLLMProvider(responses=["Staff get 24 days of annual leave [1]."])
    out = generate_cited_answer(
        "how much leave", chunks("Employees receive 24 days of annual leave."), llm
    )
    assert out.generator == "llm" and llm.calls == 1
