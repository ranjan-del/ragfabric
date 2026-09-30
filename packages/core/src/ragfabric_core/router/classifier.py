"""The router's second stage: one model call, only when the signals are not decisive.

The model sees the question and the signals and nothing else. It never sees a
chunk, so the one sentence of reasoning it returns cannot repeat anything from
a document, including one the caller is not allowed to read.
"""

from __future__ import annotations

import re
import time
from collections.abc import Collection
from dataclasses import asdict

from pydantic import BaseModel, Field

from ragfabric_core.json_contract import ContractViolation, parse_contract
from ragfabric_core.providers.base import LLMProvider, Message
from ragfabric_core.router.decision import MAX_REASONING_CHARS, QueryType, Servable
from ragfabric_core.router.signals import Signals
from ragfabric_core.strategies.base import StrategyName

CONTRACT = "router_classifier"
SYSTEM = (
    "You route questions to a retrieval strategy. You reply with one JSON object and nothing else."
)
GUIDE = """Strategies:
- traditional: meaning based search. Simple factual questions, paraphrases.
- vectorless: exact word search. Identifiers, codes, quoted phrases, names.
- graph: relationships between named people, teams, projects and places.
- agentic: comparisons, several constraints, several documents, vague questions.

Reply as {"query_type": one of simple_factual, exact_match, relationship, multi_hop,
comparison, aggregation, compound, ambiguous, "strategy": one of the allowed strategies,
"confidence": a number from 0 to 1, "reasoning": one short sentence for the user}."""
_SENTENCE_END = re.compile(r"(?<=[.!?])\s")


class ClassifierReply(BaseModel):
    query_type: QueryType
    strategy: Servable
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(min_length=1)


class ClassifierOutcome(BaseModel):
    reply: ClassifierReply | None = None
    violation: ContractViolation | None = None
    error: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    provider: str = ""
    model: str = ""
    llm_calls: int = 0
    latency_ms: int = 0


def one_sentence(text: str, limit: int = MAX_REASONING_CHARS) -> str:
    first = _SENTENCE_END.split(" ".join(text.split()), maxsplit=1)[0]
    return first if len(first) <= limit else first[: limit - 3].rstrip() + "..."


def build_prompt(question: str, signals: Signals, available: Collection[StrategyName]) -> str:
    allowed = ", ".join(str(name) for name in available if name is not StrategyName.AUTO)
    return (
        f"Question: {question}\n\nSignals: {asdict(signals)}\n\n"
        f"Allowed strategies: {allowed}\n\n{GUIDE}"
    )


def classify(
    question: str,
    signals: Signals,
    *,
    llm: LLMProvider,
    available: Collection[StrategyName],
    model: str | None = None,
) -> ClassifierOutcome:
    started = time.perf_counter()
    messages = [
        Message(role="system", content=SYSTEM),
        Message(role="user", content=build_prompt(question, signals, available)),
    ]
    try:
        completion = llm.complete(messages, model=model, max_tokens=200, temperature=0.0)
    except Exception as exc:  # noqa: BLE001  a provider can raise whatever its SDK raises
        return ClassifierOutcome(
            error=f"{type(exc).__name__}: {exc}",
            llm_calls=1,
            provider=llm.name,
            model=model or llm.default_model,
            latency_ms=int((time.perf_counter() - started) * 1000),
        )
    common = {
        "input_tokens": completion.input_tokens,
        "output_tokens": completion.output_tokens,
        "provider": completion.provider,
        "model": completion.model,
        "llm_calls": 1,
        "latency_ms": int((time.perf_counter() - started) * 1000),
    }
    parsed = parse_contract(completion.text, CONTRACT, ClassifierReply)
    if isinstance(parsed, ContractViolation):
        return ClassifierOutcome(violation=parsed, **common)
    if StrategyName(parsed.strategy) not in available:
        violation = ContractViolation(
            contract=CONTRACT,
            raw=completion.text,
            error=f"strategy {parsed.strategy} is not available for this request",
        )
        return ClassifierOutcome(violation=violation, **common)
    parsed.reasoning = one_sentence(parsed.reasoning)
    if not parsed.reasoning:
        # RouterDecision.reasoning is never empty, so a blank one cannot become a decision.
        violation = ContractViolation(
            contract=CONTRACT, raw=completion.text, error="the reasoning was empty"
        )
        return ClassifierOutcome(violation=violation, **common)
    return ClassifierOutcome(reply=parsed, **common)
