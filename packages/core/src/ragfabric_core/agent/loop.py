"""The state machine, its three ways to stop, and the trace that proves which.

Plain Python, not a graph framework. A spike built this loop both ways: the
framework version was longer, pulled in thirty-eight packages, could not carry
the model and the tools through its state, and, decisively, lost the reason the
loop stopped. A conditional edge returns the name of the next node, so the
reason has to be written somewhere else, and in the spike it silently was not,
leaving ``stop_reason=None`` on a run that had clearly stopped for a reason.
Recorded as ADR 0009.

**Every stop reason is assigned at the branch that decides it.** There is no
function anywhere that looks at a finished run and works out why it ended. That
is the whole point of the paragraph above.

The three stops:

``resolved``      every sub-question is answered or abandoned, and an abandoned
                  one carries a reason, so this is not a synonym for success.
``budget``        a cap was reached. Either the state refused a spend, or the
                  iteration cap was hit. Both are budgets; the detail says which.
``no_progress``   an iteration added no chunk id the pool did not already hold.
                  This is the one the textbook design has no answer to, and
                  without it an agent that re-fetches the same evidence will
                  spend every remaining iteration doing it again.

The repair node lives here rather than in ``nodes.py`` because it is the one
node that needs the policy, and the policy needs the node-level types. Putting
it here keeps the import one way round instead of making a cycle.
"""

from __future__ import annotations

import time

from pydantic import BaseModel, ConfigDict, Field

from ragfabric_core.agent.contracts import ContractViolation, parse_repair
from ragfabric_core.agent.nodes import (
    NodeOutcome,
    RetrievalOverride,
    ToolCall,
    _span,
    assess,
    plan,
    retrieve,
)
from ragfabric_core.agent.policy import apply_move, choose_move
from ragfabric_core.agent.state import (
    AgentState,
    BudgetExceeded,
    NodeName,
    RepairMove,
    SubQuestion,
)
from ragfabric_core.agent.tools import ToolRegistry
from ragfabric_core.providers.base import LLMProvider, Message
from ragfabric_core.strategies.base import RetrievalContext, RetrievedChunk, TraceSpan

STOP_RESOLVED = "resolved"
STOP_BUDGET = "budget"
STOP_NO_PROGRESS = "no_progress"

# Four is enough to be an agent and few enough to be bounded. Each iteration
# costs an assessment and a repair per open sub-question, and a question that
# is not converging by the fourth pass is not going to converge on the fifth.
DEFAULT_MAX_ITERATIONS = 4

REPAIR_SYSTEM = (
    "You are the repair step of a retrieval agent. "
    "You choose how to recover a failed retrieval. "
    "You reply with one JSON object and nothing else."
)

REPAIR_MOVES = """The moves available, and when each fits:

- broaden: relax the query, drop qualifiers. Use when retrieval returned nothing.
- narrow: add the missing constraint to the query. Use when retrieval returned
  plenty and none of it was on point.
- switch_strategy: swap semantic search for lexical search or back. Use when an
  identifier was missed by meaning, or a paraphrase was missed by wording.
- decompose: split this sub-question further. Use when it is still compound.
- fetch_document: pull a whole document already partly matched. Use when the
  evidence is fragmentary and the right document is identifiable.
- abandon: give up on this sub-question and say why. Use when nothing is
  working. Saying so is better than another iteration that cannot help.

Do not choose a move that has already been tried on this sub-question. It did
not work, and repeating it spends an iteration to learn that twice."""


class RepairStep(NodeOutcome):
    """One repair applied to one sub-question."""

    sub_question_index: int
    move: RepairMove
    abandoned: bool = False
    new_sub_questions: list[SubQuestion] = Field(default_factory=list)
    working: RetrievalOverride | None = None


class AgentRun(BaseModel):
    """What the loop did, in the form a strategy can report without inventing anything."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    state: AgentState
    trace: list[TraceSpan] = Field(default_factory=list)
    stop_reason: str
    stop_detail: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def retrieval_calls(self) -> int:
        """Tool invocations that actually happened, not iterations times tools."""
        return len(self.tool_calls)

    @property
    def chunks(self) -> list[RetrievedChunk]:
        """The pooled evidence, best scored first, ties broken by chunk id.

        Ordering here rather than in the strategy because the pool is a dict and
        a caller reading an unordered answer would have no way to know which
        chunk the agent thought was strongest.
        """
        return sorted(
            self.state.evidence.values(),
            key=lambda chunk: (-(chunk.score if chunk.score is not None else 0.0), chunk.chunk_id),
        )


def build_repair_prompt(
    state: AgentState,
    sub_question: SubQuestion,
    *,
    missing: str,
    returned: int,
) -> str:
    history = (
        "\n".join(
            f"- {attempt.move.value} on {attempt.query!r}: {attempt.failure}"
            for attempt in sub_question.attempts
        )
        or "- nothing has been tried yet"
    )
    return (
        f"Original question: {state.question}\n"
        f"Sub-question that is not answered: {sub_question.text}\n"
        f"Tool currently used: {sub_question.tool}\n"
        f"Chunks the last retrieval returned for it: {returned}\n"
        f"What the assessment said is missing: {missing or 'not stated'}\n\n"
        f"Already tried on this sub-question:\n{history}\n\n"
        f"{REPAIR_MOVES}\n\n"
        'Reply with JSON of the form {"move": "...", "rewritten_query": "..." or null, '
        '"why": "..."}'
    )


def repair(
    state: AgentState,
    index: int,
    *,
    llm: LLMProvider,
    working: RetrievalOverride,
    missing: str,
    returned: int,
    evidence: list[RetrievedChunk],
    origin: float | None = None,
) -> RepairStep:
    """Ask for a move, decide whether it is usable, and apply it.

    The model's proposal is a suggestion the policy is free to refuse, and a
    proposal that cannot be read at all is simply absent: the policy then picks
    from the situation. Either way a move is made, because an iteration that
    ends without changing anything is an iteration the loop will repeat.
    """
    begin = time.perf_counter()
    origin = begin if origin is None else origin
    sub_question = state.sub_questions[index]

    state.spend(NodeName.REPAIR, llm_calls=1)
    completion = llm.complete(
        [
            Message(role="system", content=REPAIR_SYSTEM),
            Message(
                role="user",
                content=build_repair_prompt(
                    state, sub_question, missing=missing, returned=returned
                ),
            ),
        ],
        temperature=0.0,
    )
    parsed = parse_repair(completion.text)

    violation: ContractViolation | None = None
    proposed: RepairMove | None = None
    rewritten: str | None = None
    why = ""
    if isinstance(parsed, ContractViolation):
        violation = parsed
    else:
        proposed = parsed.move
        rewritten = parsed.rewritten_query
        why = parsed.why

    move = choose_move(
        sub_question,
        proposed=proposed,
        has_evidence=returned > 0,
        has_document=bool(evidence),
    )
    outcome = apply_move(
        sub_question,
        move,
        working=working,
        rewritten_query=rewritten,
        missing=missing,
        why=why if move is RepairMove.ABANDON else "",
        evidence=evidence,
    )

    return RepairStep(
        span=_span(
            "repair",
            origin=origin,
            begin=begin,
            sub_question=index,
            proposed=proposed.value if proposed is not None else None,
            move=move.value,
            note=outcome.note,
            violation=violation.error if violation is not None else None,
        ),
        violation=violation,
        input_tokens=completion.input_tokens,
        output_tokens=completion.output_tokens,
        sub_question_index=index,
        move=move,
        abandoned=outcome.abandoned,
        new_sub_questions=outcome.new_sub_questions,
        working=outcome.working,
    )


def run_agent(
    question: str,
    *,
    llm: LLMProvider,
    tools: ToolRegistry,
    ctx: RetrievalContext,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    per_node_llm_calls: dict[NodeName, int] | None = None,
) -> AgentRun:
    """Plan once, then retrieve, assess and repair until one of three stops fires."""
    origin = time.perf_counter()
    state = AgentState(
        question=question,
        max_llm_calls=ctx.budget.max_llm_calls,
        per_node_llm_calls=per_node_llm_calls or {},
    )
    trace: list[TraceSpan] = []
    tool_calls: list[ToolCall] = []
    overrides: dict[int, RetrievalOverride] = {}
    tokens = [0, 0]
    stop: str | None = None
    detail = ""

    def record(outcome: NodeOutcome) -> None:
        trace.append(outcome.span)
        tokens[0] += outcome.input_tokens
        tokens[1] += outcome.output_tokens

    try:
        record(plan(state, llm=llm, tools=tools, origin=origin))
    except BudgetExceeded as exc:
        stop, detail = STOP_BUDGET, str(exc)

    while stop is None:
        if state.iterations >= max_iterations:
            stop = STOP_BUDGET
            detail = f"max_iterations of {max_iterations} reached"
            break

        state.iterations += 1
        retrieved = retrieve(state, tools=tools, ctx=ctx, overrides=overrides, origin=origin)
        record(retrieved)
        tool_calls.extend(retrieved.tool_calls)

        if not retrieved.made_progress and state.iterations > 1:
            # The first iteration is exempt: an empty first retrieval is what
            # the repair moves exist to fix, and stopping there would give the
            # agent no chance to try a different tool or a broader query. From
            # the second iteration on, no new chunk id means the repairs are
            # not reaching anything new, and another pass cannot change that.
            stop = STOP_NO_PROGRESS
            detail = f"iteration {state.iterations} added no evidence the pool did not hold"
            break

        try:
            assessed = assess(state, llm=llm, origin=origin)
        except BudgetExceeded as exc:
            stop, detail = STOP_BUDGET, str(exc)
            break
        record(assessed)

        if state.all_resolved():
            stop = STOP_RESOLVED
            detail = "every sub-question is answered or abandoned"
            break

        stop, detail = _repair_open_sub_questions(
            state,
            llm=llm,
            trace=trace,
            tokens=tokens,
            overrides=overrides,
            retrieved=retrieved,
            assessed=assessed,
            origin=origin,
        )
        if stop is None and state.all_resolved():
            stop = STOP_RESOLVED
            detail = "every sub-question is answered or abandoned"

    # Assigned at the branch above, carried here, never recomputed.
    state.stop_reason = stop
    trace.append(
        _span(
            "finalize",
            origin=origin,
            begin=time.perf_counter(),
            stop_reason=stop,
            detail=detail,
            iterations=state.iterations,
            evidence=len(state.evidence),
            answered=sum(1 for sq in state.sub_questions if sq.status.value == "answered"),
            abandoned=sum(1 for sq in state.sub_questions if sq.status.value == "abandoned"),
        )
    )
    return AgentRun(
        state=state,
        trace=trace,
        stop_reason=stop or STOP_RESOLVED,
        stop_detail=detail,
        tool_calls=tool_calls,
        input_tokens=tokens[0],
        output_tokens=tokens[1],
    )


def _repair_open_sub_questions(
    state: AgentState,
    *,
    llm: LLMProvider,
    trace: list[TraceSpan],
    tokens: list[int],
    overrides: dict[int, RetrievalOverride],
    retrieved,
    assessed,
    origin: float,
) -> tuple[str | None, str]:
    """Repair every sub-question the assessment left open.

    Returns a stop reason only when the budget refused a repair part way
    through. The repairs already applied are kept: they are real changes to the
    ledger, and discarding them would make the trace describe a run that did
    not happen.
    """
    added: list[SubQuestion] = []
    for index in list(assessed.verdicts.missing):
        sub_question = state.sub_questions[index]
        if sub_question.status.value != "open":
            continue
        working = overrides.get(index) or RetrievalOverride(query=sub_question.text, top_k=None)
        try:
            step = repair(
                state,
                index,
                llm=llm,
                working=working,
                missing=assessed.verdicts.missing.get(index, ""),
                returned=retrieved.returned_by_sub_question.get(index, 0),
                evidence=retrieved.chunks_by_sub_question.get(index, []),
                origin=origin,
            )
        except BudgetExceeded as exc:
            state.sub_questions.extend(added)
            return STOP_BUDGET, str(exc)
        trace.append(step.span)
        tokens[0] += step.input_tokens
        tokens[1] += step.output_tokens
        if step.working is not None:
            overrides[index] = step.working
        added.extend(step.new_sub_questions)
    state.sub_questions.extend(added)
    return None, ""
