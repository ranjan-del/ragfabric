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

PROMPT_VERSION = "judge-v2"
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
    # Faithfulness is not scored by the model. A small model asked for one number gave 0 to
    # answers its own reason text agreed with (judge-v1, 10 of 73 hand-labelled supported
    # answers), so it is asked for a yes or no per claim and the share is computed in code.
    "faithfulness": (
        "Split the ANSWER into its factual claims. Ignore citation markers like [1] and any "
        "statement that something could not be found. For each claim say whether the CONTEXT "
        "states it or directly implies it. Judge only against the context, not against what "
        "you know."
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

_CLAIMS_SYSTEM = (
    "You are a strict evaluator of answers produced by a retrieval system. "
    'Reply with one JSON object only: {"claims": [{"claim": "<the claim>", '
    '"supported": true or false}]}.'
)

_CLAIMS_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"claim": {"type": "string"}, "supported": {"type": "boolean"}},
                "required": ["claim", "supported"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["claims"],
    "additionalProperties": False,
}


METRICS = ("correctness", "faithfulness", "context_relevance")


def _check_metrics(metrics) -> tuple[str, ...]:
    chosen = tuple(metrics) if metrics is not None else METRICS
    unknown = [m for m in chosen if m not in METRICS]
    if unknown or not chosen:
        raise ValueError(
            f"unknown judge metric {', '.join(unknown) or '(none)'}: "
            f"expected some of {', '.join(METRICS)}"
        )
    return tuple(m for m in METRICS if m in chosen)


class JudgeScores(BaseModel):
    correctness: float | None = None
    faithfulness: float | None = None
    context_relevance: float | None = None
    reasons: dict[str, str] = Field(default_factory=dict)
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


class Judge(Protocol):
    kind: Literal["llm", "lexical", "none"]
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


def _parse_claims(text: str) -> tuple[float | None, str]:
    """Share of claims marked supported, and the unsupported ones as the reason."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    try:
        data = json.loads(match.group(0) if match else cleaned)
    except (json.JSONDecodeError, AttributeError):
        return None, f"judge reply was not JSON: {text[:80]!r}"
    claims = data.get("claims") if isinstance(data, dict) else None
    if not isinstance(claims, list) or not claims:
        return None, f"judge reply listed no claims: {text[:80]!r}"
    verdicts = []
    for item in claims:
        supported = item.get("supported") if isinstance(item, dict) else None
        if not isinstance(supported, bool):
            return None, f"judge verdict was not true or false: {str(item)[:80]!r}"
        verdicts.append((str(item.get("claim", "")), supported))
    held = sum(1 for _, ok in verdicts if ok)
    unsupported = [claim for claim, ok in verdicts if not ok]
    reason = f"{held} of {len(verdicts)} claims supported" + (
        f"; not supported: {'; '.join(unsupported)}" if unsupported else ""
    )
    return held / len(verdicts), reason


class LLMJudge:
    kind: Literal["llm"] = "llm"
    prompt_version = PROMPT_VERSION

    def __init__(
        self, llm: LLMProvider, model: str | None, metrics: tuple[str, ...] | None = None
    ) -> None:
        self.llm = llm
        self.model = model
        self.metrics = _check_metrics(metrics)

    def _ask(
        self, rubric: str, body: str, scores: JudgeScores, key: str, *, claims: bool = False
    ) -> float | None:
        messages = [
            Message(role="system", content=_CLAIMS_SYSTEM if claims else _SYSTEM),
            Message(role="user", content=f"{rubric}\n\n{body}"),
        ]
        try:
            completion = self.llm.complete(
                messages,
                model=self.model,
                max_tokens=1000 if claims else 200,
                temperature=0.0,
                json_schema=_CLAIMS_SCHEMA if claims else _SCHEMA,
            )
        except ProviderError as exc:
            scores.calls += 1
            scores.reasons[key] = f"judge call failed: {exc}"
            return None
        scores.calls += 1
        scores.input_tokens += completion.input_tokens
        scores.output_tokens += completion.output_tokens
        value, reason = (_parse_claims if claims else _parse)(completion.text)
        scores.reasons[key] = reason
        return value

    def score(
        self, question: str, expected: str, answer: str, contexts: list[ContextItem]
    ) -> JudgeScores:
        scores = JudgeScores()
        if "correctness" in self.metrics and expected.strip():
            scores.correctness = self._ask(
                RUBRICS["correctness"],
                f"QUESTION: {question}\nEXPECTED ANSWER: {expected}\nANSWER: {answer}",
                scores,
                "correctness",
            )
        if contexts:
            block = _context_block(contexts)
            if "faithfulness" in self.metrics and not _is_no_evidence(answer):
                scores.faithfulness = self._ask(
                    RUBRICS["faithfulness"],
                    f"CONTEXT:\n{block}\n\nANSWER: {answer}",
                    scores,
                    "faithfulness",
                    claims=True,
                )
            if "context_relevance" in self.metrics:
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

    def __init__(self, metrics: tuple[str, ...] | None = None) -> None:
        self.metrics = _check_metrics(metrics)

    def score(
        self, question: str, expected: str, answer: str, contexts: list[ContextItem]
    ) -> JudgeScores:
        scores = JudgeScores()
        answer_words = content_words(answer)
        expected_words = content_words(expected)
        if "correctness" in self.metrics and expected_words and answer_words:
            overlap = len(answer_words & expected_words)
            if overlap:
                p, r = overlap / len(answer_words), overlap / len(expected_words)
                scores.correctness = 2 * p * r / (p + r)
            else:
                scores.correctness = 0.0
        if "faithfulness" in self.metrics and contexts and not _is_no_evidence(answer):
            support = set().union(*(content_words(c.text) for c in contexts))
            sentences = [w for s in _SENTENCE.split(answer) if (w := content_words(s))]
            if sentences:
                held = sum(1 for w in sentences if len(w & support) / len(w) >= SUPPORT_FLOOR)
                scores.faithfulness = held / len(sentences)
        if "context_relevance" in self.metrics and contexts and expected_words:
            scores.context_relevance = sum(
                1 for c in contexts if content_words(c.text) & expected_words
            ) / len(contexts)
        return scores


class NoJudge:
    """Scores nothing: for retrieval only runs, where there is no answer to judge."""

    kind: Literal["none"] = "none"
    model = None
    prompt_version = "none"
    metrics: tuple[str, ...] = ()

    def score(
        self, question: str, expected: str, answer: str, contexts: list[ContextItem]
    ) -> JudgeScores:
        return JudgeScores()


def build_judge(
    kind: str, llm: LLMProvider, model: str | None, metrics: tuple[str, ...] | None = None
) -> Judge:
    """``auto`` is the LLM judge unless the provider is offline (D6)."""
    if kind == "lexical" or (kind == "auto" and is_offline(llm)):
        return LexicalJudge(metrics)
    if kind in ("llm", "auto"):
        if is_offline(llm):
            raise ValueError("the llm judge needs a model, and the provider is offline")
        return LLMJudge(llm, model, metrics)
    raise ValueError(f"unknown judge {kind!r}: expected auto, llm or lexical")
