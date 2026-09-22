"""The agent's nodes.

Each node is a plain function that takes the state, does one thing, mutates the
state and returns an outcome carrying a ``TraceSpan``. Nothing here decides
whether to continue; that is the loop's job, and keeping the branch out of the
nodes is what lets the stop reason be recorded at the point it is decided
rather than reconstructed afterwards.

Two node-level rules hold throughout.

**A model response can never fail the request.** Every LLM call goes through a
contract parser that returns a ``ContractViolation`` instead of raising, and
each node has a defined answer to one. For the plan that answer is a single
sub-question over the whole question, which is exactly what a plain retriever
would do and is always better than an error.

**The access filter is never reconstructed.** Tools are called with the
caller's own ``RetrievalContext``, so the filter that reaches the store is the
one the request was authorised with. The agent pools evidence across
iterations and that pool is later summarised into an answer, so a single tool
call made under the wrong filter contaminates everything downstream.
"""

from __future__ import annotations

import time
from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict, Field

from ragfabric_core.agent.contracts import ContractViolation, parse_plan
from ragfabric_core.agent.state import AgentState, NodeName, SubQuestion
from ragfabric_core.agent.tools import ToolRegistry
from ragfabric_core.providers.base import LLMProvider, Message
from ragfabric_core.strategies.base import TraceSpan

# A model asked to decompose will sometimes decompose forever. Each sub-question
# costs a retrieval on every iteration, so the list is capped: a question that
# genuinely has more than six parts is not going to be answered well by any
# number of them, and an unbounded plan turns one request into dozens of store
# queries before the budget notices.
MAX_SUB_QUESTIONS = 6

# Preference order when the plan names a tool that is not registered. Semantic
# search first because it is the tool that degrades most gracefully on a query
# it is not suited to: it returns something related, where a lexical search for
# a paraphrase can return nothing at all.
_TOOL_PREFERENCE = ("semantic_search", "lexical_search", "fetch_document")

PLAN_SYSTEM = (
    "You plan retrieval for a question answering system. "
    "You reply with one JSON object and nothing else."
)

PLAN_RULES = """Break the question into the smallest number of sub-questions that can each be
answered by one retrieval, and choose a tool for each.

Rules:
- Most questions need exactly ONE sub-question. Do not split a question that asks
  for a single fact. Splitting costs a retrieval per part and buys nothing.
- Split only when the question genuinely asks for two or more separable things,
  such as a comparison between two policies or a question joining two topics.
- Never emit more than {max_sub_questions} sub-questions.
- Each sub-question must be answerable on its own, without reading the others.

Choosing a tool:
- An identifier, an error code, a version, a file name, a quoted phrase or any
  other exact string belongs to lexical_search.
- A conceptual, paraphrased or descriptive question belongs to semantic_search.
- A request for a whole document, such as "the whole policy", belongs to
  fetch_document.
- Use only the tools listed above. Do not invent one."""


class NodeOutcome(BaseModel):
    """What a node did, in the form the loop needs to account for it."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    span: TraceSpan
    violation: ContractViolation | None = None
    input_tokens: int = 0
    output_tokens: int = 0


class PlanOutcome(NodeOutcome):
    sub_questions: list[SubQuestion] = Field(default_factory=list)


def _span(name: str, *, origin: float, begin: float, **attributes) -> TraceSpan:
    return TraceSpan(
        name=name,
        started_ms=max(0, int((begin - origin) * 1000)),
        duration_ms=max(0, int((time.perf_counter() - begin) * 1000)),
        attributes=attributes,
    )


def default_tool(tools: ToolRegistry) -> str:
    if not tools:
        raise ValueError("the agent needs at least one tool")
    for name in _TOOL_PREFERENCE:
        if name in tools:
            return name
    return next(iter(tools))


def _resolve_tool(name: str, tools: ToolRegistry) -> str:
    return name if name in tools else default_tool(tools)


def build_plan_prompt(question: str, tools: ToolRegistry) -> str:
    listed = "\n".join(f"- {tool.name}: {tool.description}" for tool in tools.values())
    return (
        f"Question: {question}\n\n"
        f"Tools available:\n{listed}\n\n"
        f"{PLAN_RULES.format(max_sub_questions=MAX_SUB_QUESTIONS)}\n\n"
        'Reply with JSON of the form {"sub_questions": '
        '[{"text": "...", "tool": "...", "why": "..."}]}'
    )


def plan(
    state: AgentState,
    *,
    llm: LLMProvider,
    tools: ToolRegistry,
    origin: float | None = None,
) -> PlanOutcome:
    """Decompose the question and route each part to a tool, in one call.

    This is the merged ``analyze`` plus ``plan_need`` of the textbook design.
    They were two calls doing adjacent work: understanding the question and
    choosing how to answer it are the same judgement, and splitting them bought
    a second failure point on every request.
    """
    begin = time.perf_counter()
    origin = begin if origin is None else origin
    state.spend(NodeName.PLAN, llm_calls=1)

    completion = llm.complete(
        [
            Message(role="system", content=PLAN_SYSTEM),
            Message(role="user", content=build_plan_prompt(state.question, tools)),
        ],
        temperature=0.0,
    )
    parsed = parse_plan(completion.text)

    violation: ContractViolation | None = None
    if isinstance(parsed, ContractViolation):
        violation = parsed
        sub_questions = [SubQuestion(text=state.question, tool=default_tool(tools))]
    else:
        sub_questions = _dedupe(
            SubQuestion(text=planned.text.strip(), tool=_resolve_tool(planned.tool, tools))
            for planned in parsed.sub_questions
        )[:MAX_SUB_QUESTIONS]

    state.sub_questions = sub_questions
    return PlanOutcome(
        span=_span(
            "plan",
            origin=origin,
            begin=begin,
            sub_questions=len(sub_questions),
            tools=",".join(sorted({sq.tool for sq in sub_questions})),
            fallback=violation is not None,
            violation=violation.error if violation is not None else None,
        ),
        violation=violation,
        sub_questions=sub_questions,
        input_tokens=completion.input_tokens,
        output_tokens=completion.output_tokens,
    )


def _dedupe(sub_questions: Iterable[SubQuestion]) -> list[SubQuestion]:
    """Collapse sub-questions that differ only in case or surrounding space.

    A model that restates the same part twice would otherwise double every
    retrieval for it and, worse, make the evidence pool look busier than the
    agent's understanding actually is.
    """
    seen: set[str] = set()
    kept: list[SubQuestion] = []
    for sq in sub_questions:
        key = " ".join(sq.text.lower().split())
        if key in seen:
            continue
        seen.add(key)
        kept.append(sq)
    return kept
