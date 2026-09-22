"""What the agent asks a model for, and what it will accept back.

Every decision the agent delegates to a model comes back as JSON validated
against a schema here. There is no native tool calling involved, deliberately:
``LLMProvider`` exposes only ``complete`` and ``stream``, and building a
tool-calling surface would be a second subsystem that could not be exercised
without paid API credit. JSON contracts work on every provider, including a
3B model running locally, and they let the scripted test double drive every
branch of the loop offline.

The parsers are deliberately forgiving about *packaging* and strict about
*content*. Small models wrap JSON in a fenced block, preface it with a sentence
of agreement, or explain themselves afterwards. None of that is a real failure,
so it is tolerated. A missing field, an empty plan or an invented repair move
is a real failure, and it returns a ``ContractViolation`` rather than raising,
so the caller decides whether to retry with a corrective instruction or fall
back to a safe default. A malformed response must never crash a request and
must never be silently counted as success.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from ragfabric_core.agent.state import RepairMove

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


class ContractViolation(BaseModel):
    """A model response that could not be honoured.

    ``raw`` is kept because a violation that does not say what the model
    actually returned cannot be diagnosed later from a trace.
    """

    contract: str
    raw: str
    error: str


class PlannedSubQuestion(BaseModel):
    text: str = Field(min_length=1)
    tool: str = Field(min_length=1)
    why: str = ""


class PlanResponse(BaseModel):
    sub_questions: list[PlannedSubQuestion] = Field(min_length=1)


class Verdict(BaseModel):
    sub_question: str = Field(min_length=1)
    answered: bool
    missing: str | None = None


class AssessResponse(BaseModel):
    verdicts: list[Verdict]


class RepairResponse(BaseModel):
    move: RepairMove
    rewritten_query: str | None = None
    why: str = ""


def _candidates(text: str) -> list[str]:
    """Every substring of the response that might be the JSON object.

    Tried in order: fenced code blocks first, because when a model uses one it
    is almost always the real payload, then the widest brace-balanced span.
    """
    found = [block.strip() for block in _FENCE.findall(text)]
    depth = 0
    start = -1
    for index, char in enumerate(text):
        if char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                found.append(text[start : index + 1])
                start = -1
    return found


def _load(text: str) -> tuple[dict[str, Any] | None, str]:
    for candidate in _candidates(text):
        try:
            loaded = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(loaded, dict):
            return loaded, ""
    return None, "no JSON object found in the response"


def _parse(text: str, contract: str, model: type[BaseModel]) -> Any:
    payload, error = _load(text)
    if payload is None:
        return ContractViolation(contract=contract, raw=text, error=error)
    try:
        return model.model_validate(payload)
    except ValidationError as exc:
        return ContractViolation(contract=contract, raw=text, error=str(exc))


def parse_plan(text: str) -> PlanResponse | ContractViolation:
    """A plan with no sub-questions is refused.

    An empty plan leaves the loop nothing to retrieve for, and it would then
    report no progress on the next pass, which hides the real problem.
    """
    return _parse(text, "plan", PlanResponse)


def parse_assess(text: str) -> AssessResponse | ContractViolation:
    return _parse(text, "assess", AssessResponse)


def parse_repair(text: str) -> RepairResponse | ContractViolation:
    """An invented move is refused by the schema before it reaches the policy."""
    return _parse(text, "repair", RepairResponse)
