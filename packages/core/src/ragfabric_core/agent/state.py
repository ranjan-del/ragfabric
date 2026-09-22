"""The agent's working state.

Three ideas carry the whole design.

**The sub-question ledger.** A question is decomposed into sub-questions, and
each one owns its status, the tool it is being answered with, and the history of
what has already been tried on it. Judging evidence for the question as a whole
would make decomposition pointless, because the loop could never tell which part
is still missing and would re-retrieve everything on every pass.

**The evidence pool is keyed by chunk id.** Retrieving the same chunk twice adds
nothing, and ``add_evidence`` returns only the ids that were genuinely new. That
return value is what progress detection reads: an iteration that adds no new ids
has not progressed, however many rows the stores returned.

**Spending is refused, not merely recorded.** ``spend`` raises before the counter
moves once a cap would be crossed, and there are per-node caps as well as a
global one, because a global cap alone lets a single runaway node consume the
whole budget before the other nodes ever run.
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

from ragfabric_core.strategies.base import RetrievedChunk

# How hard the assess node judges evidence. Declared here, in the module both
# the nodes and the configuration already import, so the two cannot drift into
# two spellings of the same setting.
AssessStrictness = Literal["strict", "lenient"]

DEFAULT_ASSESS_STRICTNESS: AssessStrictness = "strict"


class BudgetExceeded(Exception):
    """Raised when a spend would cross a global or per-node cap.

    The loop catches this and finalizes with what it has, which is why the
    message names the cap that tripped: it becomes the stop reason the caller
    sees, and an unexplained stop is the failure this phase exists to avoid.
    """


class NodeName(StrEnum):
    PLAN = "plan"
    RETRIEVE = "retrieve"
    ASSESS = "assess"
    REPAIR = "repair"
    GENERATE = "generate"
    FINALIZE = "finalize"


class SubQuestionStatus(StrEnum):
    OPEN = "open"
    ANSWERED = "answered"
    ABANDONED = "abandoned"


class RepairMove(StrEnum):
    """The complete set of moves the agent may make when a sub-question fails.

    A fixed enum rather than free text, because the move is chosen by a model.
    A weak model cannot invent an action that does not exist here; the worst it
    can do is pick a valid move that is a poor fit, which the next iteration can
    correct. That property is what lets this design run on a small local model.
    """

    BROADEN = "broaden"
    NARROW = "narrow"
    SWITCH_STRATEGY = "switch_strategy"
    DECOMPOSE = "decompose"
    FETCH_DOCUMENT = "fetch_document"
    ABANDON = "abandon"


class Attempt(BaseModel):
    """One repair already made on a sub-question, and why it did not work.

    The failure text is kept because "try something different" is a uniqueness
    check, not learning. The next repair is chosen knowing what already failed.
    """

    move: RepairMove
    query: str
    failure: str


class SubQuestion(BaseModel):
    text: str
    tool: str
    status: SubQuestionStatus = SubQuestionStatus.OPEN
    attempts: list[Attempt] = Field(default_factory=list)
    reason: str | None = None

    def mark_answered(self) -> None:
        self.status = SubQuestionStatus.ANSWERED

    def abandon(self, reason: str) -> None:
        """Give up on this sub-question, on the record.

        A reason is required. An abandoned sub-question with no explanation is a
        silent failure, and the caller would see a gap in the answer with nothing
        saying why it is there.
        """
        if not reason.strip():
            raise ValueError("abandoning a sub-question requires a reason")
        self.status = SubQuestionStatus.ABANDONED
        self.reason = reason

    def record_attempt(self, *, move: RepairMove, query: str, failure: str) -> None:
        self.attempts.append(Attempt(move=move, query=query, failure=failure))

    def has_tried(self, move: RepairMove) -> bool:
        return any(attempt.move is move for attempt in self.attempts)

    def tried_moves(self) -> set[RepairMove]:
        return {attempt.move for attempt in self.attempts}


class AgentState(BaseModel):
    question: str
    sub_questions: list[SubQuestion] = Field(default_factory=list)
    evidence: dict[int, RetrievedChunk] = Field(default_factory=dict)

    max_llm_calls: int = Field(default=12, ge=0)
    per_node_llm_calls: dict[NodeName, int] = Field(default_factory=dict)
    llm_calls: int = Field(default=0, ge=0)
    node_llm_calls: dict[NodeName, int] = Field(default_factory=dict)

    iterations: int = Field(default=0, ge=0)
    stop_reason: str | None = None

    def spend(self, node: NodeName, *, llm_calls: int = 0) -> None:
        """Account for work about to be done, refusing it if a cap would break.

        Checked before the counters move, so a refused spend leaves the state
        exactly as it was and the caller can finalize on the evidence it already
        has rather than on a half-applied step.
        """
        if self.llm_calls + llm_calls > self.max_llm_calls:
            raise BudgetExceeded(
                f"max_llm_calls of {self.max_llm_calls} reached at node {node.value}"
            )
        node_cap = self.per_node_llm_calls.get(node)
        if node_cap is not None:
            already = self.node_llm_calls.get(node, 0)
            if already + llm_calls > node_cap:
                raise BudgetExceeded(f"per-node cap of {node_cap} reached at node {node.value}")
        self.llm_calls += llm_calls
        self.node_llm_calls[node] = self.node_llm_calls.get(node, 0) + llm_calls

    def add_evidence(self, chunks: Iterable[RetrievedChunk]) -> set[int]:
        """Pool chunks by id and return only the ids that were not already held.

        The return value is the progress signal. An empty set means this
        retrieval learned nothing new, which is the condition the loop treats as
        being stuck.
        """
        new_ids: set[int] = set()
        for chunk in chunks:
            if chunk.chunk_id in self.evidence:
                continue
            self.evidence[chunk.chunk_id] = chunk
            new_ids.add(chunk.chunk_id)
        return new_ids

    def open_sub_questions(self) -> list[SubQuestion]:
        return [sq for sq in self.sub_questions if sq.status is SubQuestionStatus.OPEN]

    def all_resolved(self) -> bool:
        return bool(self.sub_questions) and not self.open_sub_questions()
