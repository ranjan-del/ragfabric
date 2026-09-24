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

from pydantic import BaseModel, Field

from ragfabric_core.agent.state import RepairMove
from ragfabric_core.json_contract import ContractViolation, parse_contract


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


def parse_plan(text: str) -> PlanResponse | ContractViolation:
    """A plan with no sub-questions is refused.

    An empty plan leaves the loop nothing to retrieve for, and it would then
    report no progress on the next pass, which hides the real problem.
    """
    return parse_contract(text, "plan", PlanResponse)


def parse_assess(text: str) -> AssessResponse | ContractViolation:
    return parse_contract(text, "assess", AssessResponse)


def parse_repair(text: str) -> RepairResponse | ContractViolation:
    """An invented move is refused by the schema before it reaches the policy."""
    return parse_contract(text, "repair", RepairResponse)
