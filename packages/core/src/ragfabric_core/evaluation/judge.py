"""Judges for the generation metrics (D6).

``LLMJudge`` asks a model three fixed rubric questions (correctness,
faithfulness, context relevance) at temperature 0 and expects one JSON object
per answer. Anything it cannot read as a score between 0 and 1 is recorded as
``None`` with the reason; a judge that is not understood is not guessed at.

``LexicalJudge`` is deterministic word overlap. It keeps CI and offline runs
reproducible, and every report names it, so its numbers are never mistaken
for the rubric's.
"""

from __future__ import annotations

import json
import re
from typing import Literal, Protocol

from pydantic import BaseModel, Field

from ragfabric_core.evaluation.metrics import _SENTENCE, SUPPORT_FLOOR, content_words
from ragfabric_core.evaluation.target import ContextItem
from ragfabric_core.generate.contract import NO_EVIDENCE
from ragfabric_core.providers.base import LLMProvider, Message, ProviderError, is_offline

PROMPT_VERSION = "judge-v1"
LEXICAL_VERSION = "lexical-v1"

_SYSTEM = (
    "You are a strict evaluator of answers produced by a retrieval system. "
    'Reply with one JSON object only: {"score": <number from 0 to 1>, "reason": "<one sentence>"}.'
)

RUBRICS = {
    "correctness": (
        "Score how well the ANSWER states the facts in the EXPECTED ANSWER. 1 means every fact "
        "in the expected answer is stated correctly; 0 means none is, or the answer contradicts "
        "it. Ignore wording, citation markers like [1], and extra correct detail."
    ),
    "faithfulness": (
        "Score the share of claims in the ANSWER that the CONTEXT supports. 1 means every claim "
        "is supported by the context; 0 means none is. Judge only against the context, not "
        "against what you know."
    ),
    "context_relevance": (
        "Score the share of the numbered CONTEXT passages that contain information needed to "
        "answer the QUESTION. 1 means every passage is needed; 0 means none is."
    ),
}

_SCHEMA = {
    "type": "object",
    "properties": {"score": {"type": "number"}, "reason": {"type": "string"}},
    "required": ["score", "reason"],
    # Strict structured output (OpenAI) refuses a schema that allows extra keys.
    "additionalProperties": False,
}


class JudgeScores(BaseModel):
    correctness: float | None = None
    faithfulness: float | None = None
    context_relevance: float | None = None
    reasons: dict[str, str] = Field(default_factory=dict)
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


class Judge(Protocol):
    kind: Literal["llm", "lexical"]
    model: str | None
    prompt_version: str

    def score(
        self, question: str, expected: str, answer: str, contexts: list[ContextItem]
    ) -> JudgeScores: ...


def _is_no_evidence(answer: str) -> bool:
    return NO_EVIDENCE in answer.lower()


def _context_block(contexts: list[ContextItem]) -> str:
    return "\n".join(f"[{c.rank}] ({c.document}) {c.text}" for c in contexts)


def _parse(text: str) -> tuple[float | None, str]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    try:
        data = json.loads(match.group(0) if match else cleaned)
    except (json.JSONDecodeError, AttributeError):
        return None, f"judge reply was not JSON: {text[:80]!r}"
    score = data.get("score") if isinstance(data, dict) else None
    reason = str(data.get("reason", "")) if isinstance(data, dict) else ""
    if isinstance(score, bool) or not isinstance(score, int | float):
        return None, f"judge score was not a number: {score!r}"
    if not 0.0 <= float(score) <= 1.0:
        return None, f"judge score {score} is outside 0 to 1"
    return float(score), reason


class LLMJudge:
    kind: Literal["llm"] = "llm"
    prompt_version = PROMPT_VERSION

    def __init__(self, llm: LLMProvider, model: str | None) -> None:
        self.llm = llm
        self.model = model

    def _ask(self, rubric: str, body: str, scores: JudgeScores, key: str) -> float | None:
        messages = [
            Message(role="system", content=_SYSTEM),
            Message(role="user", content=f"{rubric}\n\n{body}"),
        ]
        try:
            completion = self.llm.complete(
                messages, model=self.model, max_tokens=200, temperature=0.0, json_schema=_SCHEMA
            )
        except ProviderError as exc:
            scores.calls += 1
            scores.reasons[key] = f"judge call failed: {exc}"
            return None
        scores.calls += 1
        scores.input_tokens += completion.input_tokens
        scores.output_tokens += completion.output_tokens
        value, reason = _parse(completion.text)
        scores.reasons[key] = reason
        return value

    def score(
        self, question: str, expected: str, answer: str, contexts: list[ContextItem]
    ) -> JudgeScores:
        scores = JudgeScores()
        if expected.strip():
            scores.correctness = self._ask(
                RUBRICS["correctness"],
                f"QUESTION: {question}\nEXPECTED ANSWER: {expected}\nANSWER: {answer}",
                scores,
                "correctness",
            )
        if contexts:
            block = _context_block(contexts)
            if not _is_no_evidence(answer):
                scores.faithfulness = self._ask(
                    RUBRICS["faithfulness"],
                    f"CONTEXT:\n{block}\n\nANSWER: {answer}",
                    scores,
                    "faithfulness",
                )
            scores.context_relevance = self._ask(
                RUBRICS["context_relevance"],
                f"QUESTION: {question}\n\nCONTEXT:\n{block}",
                scores,
                "context_relevance",
            )
        return scores


class LexicalJudge:
    kind: Literal["lexical"] = "lexical"
    model = None
    prompt_version = LEXICAL_VERSION

    def score(
        self, question: str, expected: str, answer: str, contexts: list[ContextItem]
    ) -> JudgeScores:
        scores = JudgeScores()
        answer_words = content_words(answer)
        expected_words = content_words(expected)
        if expected_words and answer_words:
            overlap = len(answer_words & expected_words)
            if overlap:
                p, r = overlap / len(answer_words), overlap / len(expected_words)
                scores.correctness = 2 * p * r / (p + r)
            else:
                scores.correctness = 0.0
        if contexts and not _is_no_evidence(answer):
            support = set().union(*(content_words(c.text) for c in contexts))
            sentences = [w for s in _SENTENCE.split(answer) if (w := content_words(s))]
            if sentences:
                held = sum(1 for w in sentences if len(w & support) / len(w) >= SUPPORT_FLOOR)
                scores.faithfulness = held / len(sentences)
        if contexts and expected_words:
            scores.context_relevance = sum(
                1 for c in contexts if content_words(c.text) & expected_words
            ) / len(contexts)
        return scores


def build_judge(kind: str, llm: LLMProvider, model: str | None) -> Judge:
    """``auto`` is the LLM judge unless the provider is offline (D6)."""
    if kind == "lexical" or (kind == "auto" and is_offline(llm)):
        return LexicalJudge()
    if kind in ("llm", "auto"):
        if is_offline(llm):
            raise ValueError("the llm judge needs a model, and the provider is offline")
        return LLMJudge(llm, model)
    raise ValueError(f"unknown judge {kind!r}: expected auto, llm or lexical")
