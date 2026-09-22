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
would do and is always better than an error. For the assessment it is to leave
every sub-question open, because a model that could not be read has not said
anything is answered.

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

from ragfabric_core.agent.contracts import ContractViolation, parse_assess, parse_plan
from ragfabric_core.agent.state import AgentState, NodeName, SubQuestion, SubQuestionStatus
from ragfabric_core.agent.tools import ToolRegistry
from ragfabric_core.providers.base import LLMProvider, Message
from ragfabric_core.strategies.base import RetrievalContext, RetrievedChunk, TraceSpan

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

ASSESS_SYSTEM = (
    "You judge whether retrieved evidence answers a question. "
    "You reply with one JSON object and nothing else."
)

ASSESS_RUBRIC = """For each sub-question, decide whether the evidence above answers it.

The rubric is strict, and being wrong in the permissive direction is the worse
error. A sub-question counts as answered ONLY if the answer can be read
directly out of the evidence text. Specifically:

- Evidence that is about the same topic but does not state the answer is NOT
  answered. Related is not answered.
- Evidence that implies the answer, or that would let a knowledgeable reader
  guess it, is NOT answered.
- Evidence that answers a different question about the same subject is NOT
  answered.
- If you are unsure, answer false. An honest gap is better than a confident
  wrong answer.

When a sub-question is not answered, say in "missing" exactly what is absent,
in one short phrase. That phrase chooses the next retrieval, so "the number of
retries allowed" is useful and "more information" is not."""


class NodeOutcome(BaseModel):
    """What a node did, in the form the loop needs to account for it."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    span: TraceSpan
    violation: ContractViolation | None = None
    input_tokens: int = 0
    output_tokens: int = 0


class PlanOutcome(NodeOutcome):
    sub_questions: list[SubQuestion] = Field(default_factory=list)


class ToolCall(BaseModel):
    """One tool invocation, recorded so the counters report what happened."""

    tool: str
    query: str
    sub_question_index: int
    returned: int


class RetrieveOutcome(NodeOutcome):
    new_chunk_ids: set[int] = Field(default_factory=set)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    returned_by_sub_question: dict[int, int] = Field(default_factory=dict)

    @property
    def made_progress(self) -> bool:
        """True only when this iteration put evidence in the pool it did not hold.

        Read the definition carefully, because the obvious alternative is wrong.
        Progress is not "the tools returned something": a tool that returns the
        same three chunks on every call returns something every time and teaches
        the agent nothing. Progress is new chunk IDS, which is why this reads
        the set returned by ``AgentState.add_evidence`` and never a count of
        rows.
        """
        return bool(self.new_chunk_ids)


class Verdicts(BaseModel):
    """The assessment, indexed by position in ``state.sub_questions``.

    Indexes rather than text, because the repair has to act on the ledger entry
    and matching a model's paraphrase of a sub-question back to the real one at
    that point is a second place for the same mistake.
    """

    answered: list[int] = Field(default_factory=list)
    missing: dict[int, str] = Field(default_factory=dict)
    unjudged: list[int] = Field(default_factory=list)


class AssessOutcome(NodeOutcome):
    verdicts: Verdicts = Field(default_factory=Verdicts)


class RetrievalOverride(BaseModel):
    """The working form of a sub-question after the repair policy changed it.

    The sub-question's own ``text`` stays as the plan wrote it, because that is
    what the caller is shown in the per sub-question report. The query actually
    sent to a tool lives here, so a broadened or narrowed query never rewrites
    the record of what was asked.
    """

    query: str
    top_k: int | None = None
    document_id: int | None = None


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


def retrieve(
    state: AgentState,
    *,
    tools: ToolRegistry,
    ctx: RetrievalContext,
    overrides: dict[int, RetrievalOverride] | None = None,
    origin: float | None = None,
) -> RetrieveOutcome:
    """Run each open sub-question's tool and pool what comes back.

    Only open sub-questions are retrieved for. That is the whole reason the
    ledger exists: once a part of the question is answered, spending another
    store query on it is what judging evidence globally looks like from the
    inside, and it is what makes decomposition pointless.

    The returned outcome carries the ids that were genuinely new. An iteration
    that produced none of those has not progressed, however many rows came back,
    and the loop stops rather than spending the rest of the budget re-reading
    what it already has.
    """
    begin = time.perf_counter()
    origin = begin if origin is None else origin
    overrides = overrides or {}

    calls: list[ToolCall] = []
    returned_by_sub_question: dict[int, int] = {}
    harvested: list[RetrievedChunk] = []

    open_indexes = _open_indexes(state)
    for index in open_indexes:
        sq = state.sub_questions[index]
        override = overrides.get(index)
        query = override.query if override is not None else sq.text
        tool = tools.get(sq.tool)
        if tool is None:
            # A repair can point a sub-question at a tool this deployment does
            # not run. Returning nothing lets the next repair try something
            # else; raising would fail a request over a recoverable choice.
            continue
        chunks = tool.run(query, _with_override(ctx, override))
        harvested.extend(chunks)
        returned_by_sub_question[index] = len(chunks)
        calls.append(
            ToolCall(tool=sq.tool, query=query, sub_question_index=index, returned=len(chunks))
        )

    new_ids = state.add_evidence(harvested)
    return RetrieveOutcome(
        span=_span(
            "retrieve",
            origin=origin,
            begin=begin,
            tool_calls=len(calls),
            returned=len(harvested),
            new_chunks=len(new_ids),
            pooled=len(state.evidence),
            progress=bool(new_ids),
        ),
        new_chunk_ids=new_ids,
        tool_calls=calls,
        returned_by_sub_question=returned_by_sub_question,
    )


def _open_indexes(state: AgentState) -> list[int]:
    return [
        index for index, sq in enumerate(state.sub_questions) if sq.status is SubQuestionStatus.OPEN
    ]


def _with_override(ctx: RetrievalContext, override: RetrievalOverride | None) -> RetrievalContext:
    """Apply a repair's ``top_k`` change without rebuilding the access filter.

    ``model_copy`` carries the principal and the filter over untouched, so
    broadening a search widens what is searched and never what the caller is
    allowed to see. Constructing a fresh ``RetrievalContext`` here would make
    that a matter of remembering to copy a field (ADR 0003).
    """
    if override is None or override.top_k is None or override.top_k == ctx.params.top_k:
        return ctx
    params = ctx.params.model_copy(update={"top_k": override.top_k})
    return ctx.model_copy(update={"params": params})


def build_assess_prompt(state: AgentState, open_indexes: list[int]) -> str:
    evidence = "\n\n".join(f"[{chunk.chunk_id}] {chunk.text}" for chunk in state.evidence.values())
    listed = "\n".join(f"- {state.sub_questions[index].text}" for index in open_indexes)
    return (
        f"Original question: {state.question}\n\n"
        f"Evidence retrieved so far:\n{evidence or '(nothing was retrieved)'}\n\n"
        f"Sub-questions to judge:\n{listed}\n\n"
        f"{ASSESS_RUBRIC}\n\n"
        'Reply with JSON of the form {"verdicts": [{"sub_question": "...", '
        '"answered": true, "missing": null}]}'
    )


def assess(
    state: AgentState,
    *,
    llm: LLMProvider,
    origin: float | None = None,
) -> AssessOutcome:
    """Judge the pooled evidence per open sub-question, pessimistically.

    Judging per sub-question rather than for the question as a whole is what
    makes the next iteration cheap: the agent chases only what is still
    missing, instead of re-retrieving everything because "the answer is not
    complete yet".

    Two pessimistic readings are enforced here rather than merely requested in
    the prompt, because a rubric in a prompt is a request and this is the node
    where being wrong is most expensive.

    A verdict that says answered while also naming something missing is treated
    as not answered. Small models produce exactly that contradiction, and the
    safe reading of it is the one that keeps looking.

    A sub-question the model did not judge at all stays open. Silence is not
    consent, and an unjudged sub-question quietly marked answered is how a gap
    reaches the caller wearing a citation.
    """
    begin = time.perf_counter()
    origin = begin if origin is None else origin

    open_indexes = _open_indexes(state)
    if not open_indexes:
        # Nothing to judge costs nothing. The budget exists to be spent on work.
        return AssessOutcome(
            span=_span("assess", origin=origin, begin=begin, judged=0, answered=0, skipped=True)
        )

    state.spend(NodeName.ASSESS, llm_calls=1)
    completion = llm.complete(
        [
            Message(role="system", content=ASSESS_SYSTEM),
            Message(role="user", content=build_assess_prompt(state, open_indexes)),
        ],
        temperature=0.0,
    )
    parsed = parse_assess(completion.text)

    verdicts = Verdicts()
    violation: ContractViolation | None = None
    if isinstance(parsed, ContractViolation):
        violation = parsed
        verdicts.unjudged = list(open_indexes)
    else:
        by_text = {_match_key(state.sub_questions[i].text): i for i in open_indexes}
        judged: set[int] = set()
        for verdict in parsed.verdicts:
            index = by_text.get(_match_key(verdict.sub_question))
            if index is None:
                # A verdict on a sub-question that was never asked cannot close
                # one that was. Matching it to the nearest real sub-question
                # would be guessing on the model's behalf.
                continue
            judged.add(index)
            missing = (verdict.missing or "").strip()
            if verdict.answered and not missing:
                state.sub_questions[index].mark_answered()
                verdicts.answered.append(index)
            else:
                verdicts.missing[index] = missing or "the evidence does not state the answer"
        verdicts.unjudged = [index for index in open_indexes if index not in judged]

    for index in verdicts.unjudged:
        verdicts.missing.setdefault(index, "the assessment did not judge this sub-question")

    return AssessOutcome(
        span=_span(
            "assess",
            origin=origin,
            begin=begin,
            judged=len(open_indexes),
            answered=len(verdicts.answered),
            unjudged=len(verdicts.unjudged),
            skipped=False,
            violation=violation.error if violation is not None else None,
        ),
        violation=violation,
        verdicts=verdicts,
        input_tokens=completion.input_tokens,
        output_tokens=completion.output_tokens,
    )


def _match_key(text: str) -> str:
    """Normalise a sub-question for matching a verdict back to the ledger.

    Case, spacing and a trailing question mark are all things a model changes
    without meaning anything by it. Anything beyond that is a different
    sub-question and is deliberately not matched.
    """
    return " ".join(text.lower().split()).rstrip("?.")
