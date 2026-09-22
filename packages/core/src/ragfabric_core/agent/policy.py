"""The repair policy: what the agent does when a sub-question is not answered.

The textbook agentic design repairs failure exactly one way, by rewriting the
query. That is a retry loop with a vocabulary: every failure gets the same
response, so a failure that a different wording cannot fix is never fixed. This
module is the alternative. Six moves, each doing something structurally
different to the sub-question, and a choice between them that is made from what
has already been tried.

**Choosing.** The model proposes a move and the policy decides whether that
proposal is usable. Two things can make it unusable: it has already been tried
on this sub-question, or it cannot do anything here (there is nothing to split,
or no document to fetch). Either way the policy falls back to a situational
order, and when nothing is left it abandons with a reason. The proposal is
never rejected for being unexpected, because the enum already made an invented
move impossible.

**Not repeating.** A move already tried on a sub-question is not offered again.
Without that guard an agent will alternate broaden and narrow until the budget
is gone, and every iteration will look like work. The guard is deliberately
strict: the attempt history is per sub-question, so a move being spent here
says nothing about another sub-question, and a sub-question that has genuinely
changed has been replaced by new ones (that is what decompose does) with an
empty history of their own.

**Abandoning is a move, not a failure.** When nothing is left, saying so with a
reason is the honest outcome, and it is strictly better than a loop that keeps
paying for iterations it cannot use.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, Field

from ragfabric_core.agent.nodes import RetrievalOverride
from ragfabric_core.agent.state import RepairMove, SubQuestion
from ragfabric_core.strategies.base import RetrievedChunk

# Order tried when the model proposes nothing usable.
#
# Nothing came back at all: the query is probably too specific for the corpus,
# so widen it, then try the other retrieval shape, then a smaller question.
# Narrow is nearly last here because narrowing a query that already returned
# nothing is how an agent talks itself into an empty result twice.
MOVES_WHEN_NOTHING_CAME_BACK = (
    RepairMove.BROADEN,
    RepairMove.SWITCH_STRATEGY,
    RepairMove.DECOMPOSE,
    RepairMove.FETCH_DOCUMENT,
    RepairMove.NARROW,
)

# Evidence came back and none of it answers the sub-question: the retrieval is
# in the right region and pointed slightly wrong, so add the missing constraint
# first, and only widen as a last resort.
MOVES_WHEN_EVIDENCE_IS_OFF_POINT = (
    RepairMove.NARROW,
    RepairMove.SWITCH_STRATEGY,
    RepairMove.FETCH_DOCUMENT,
    RepairMove.DECOMPOSE,
    RepairMove.BROADEN,
)

# StrategyParams caps top_k at 100, so broadening stops there rather than
# producing a value the context would reject.
MAX_TOP_K = 100

# What narrowing cuts back to. A narrowed query is a more specific one, and
# asking for twenty candidates for it re-admits the noise the narrowing was
# meant to remove.
NARROW_TOP_K = 5

_PAIRED_TOOLS = {
    "semantic_search": "lexical_search",
    "lexical_search": "semantic_search",
    # A sub-question already pointed at a whole document has no lexical or
    # semantic counterpart to flip to, so it goes back to searching.
    "fetch_document": "semantic_search",
}

# Words that make a query more specific without making it more findable. A
# corpus does not use the word "exact", the asker does.
_QUALIFIERS = frozenset(
    {
        "exact",
        "exactly",
        "precise",
        "precisely",
        "specific",
        "specifically",
        "current",
        "currently",
        "official",
        "actual",
        "actually",
    }
)

# Prepositions that usually open a trailing restriction ("for the payments API",
# "in production"). Dropping from the first of these widens the query while
# keeping its subject.
_TRAILING_PREPOSITIONS = (" for ", " in ", " during ", " under ", " within ", " when ")

# Conjunctions a compound sub-question is split on. Ordered longest first so
# "compared to" wins over the bare comma inside it.
_SPLITTERS = (" compared to ", " versus ", " vs ", " and ", "; ")

# The shortest a decomposed part may be. Below this the split is almost always
# inside a phrase ("terms and conditions") rather than between two questions.
_MIN_PART_WORDS = 3

_STOPWORDS = frozenset(
    {"a", "an", "the", "is", "are", "was", "were", "of", "to", "what", "which", "does", "do"}
)


class RepairOutcome(BaseModel):
    """What a repair actually did, in the form the loop needs to apply it."""

    move: RepairMove
    working: RetrievalOverride
    new_sub_questions: list[SubQuestion] = Field(default_factory=list)
    abandoned: bool = False
    note: str = ""


def can_split(text: str) -> bool:
    return len(_split(text)) > 1


def choose_move(
    sub_question: SubQuestion,
    *,
    proposed: RepairMove | None,
    has_evidence: bool,
    has_document: bool | None = None,
) -> RepairMove:
    """Pick the move to make, honouring the model's proposal when it is usable.

    ``has_document`` defaults to ``has_evidence`` because the only document the
    agent can sensibly fetch is one it has already matched part of. Passing it
    explicitly is how the loop says "there is evidence, but none of it carries a
    document worth pulling whole".
    """
    if has_document is None:
        has_document = has_evidence

    def usable(move: RepairMove) -> bool:
        if sub_question.has_tried(move):
            return False
        if move is RepairMove.DECOMPOSE:
            return can_split(sub_question.text)
        if move is RepairMove.FETCH_DOCUMENT:
            return has_document
        return True

    if proposed is RepairMove.ABANDON and sub_question.attempts:
        # Honoured only once something has actually been tried. A model that
        # gives up on the first iteration would otherwise turn the agent into a
        # plain retriever that reports a gap it never looked for.
        return RepairMove.ABANDON
    if proposed is not None and proposed is not RepairMove.ABANDON and usable(proposed):
        return proposed

    order = MOVES_WHEN_EVIDENCE_IS_OFF_POINT if has_evidence else MOVES_WHEN_NOTHING_CAME_BACK
    for move in order:
        if usable(move):
            return move
    return RepairMove.ABANDON


def apply_move(
    sub_question: SubQuestion,
    move: RepairMove,
    *,
    working: RetrievalOverride,
    rewritten_query: str | None = None,
    missing: str = "",
    why: str = "",
    evidence: Sequence[RetrievedChunk] = (),
) -> RepairOutcome:
    """Carry out a move, recording the attempt that led to it.

    The attempt is recorded before the sub-question is changed, so the history
    holds the query that actually failed and the reason it failed, not the
    replacement. "Try something different" is a uniqueness check; keeping the
    failure is what makes the next choice better than the last.
    """
    failure = missing or why or "the evidence did not answer this sub-question"

    if move is RepairMove.ABANDON:
        reason = why.strip() or _abandon_reason(sub_question, failure)
        sub_question.abandon(reason)
        return RepairOutcome(move=move, working=working, abandoned=True, note=reason)

    sub_question.record_attempt(move=move, query=working.query, failure=failure)

    if move is RepairMove.BROADEN:
        query = (rewritten_query or "").strip() or _broaden(working.query)
        top_k = min(MAX_TOP_K, max((working.top_k or 5) * 2, (working.top_k or 5) + 5))
        return RepairOutcome(
            move=move,
            working=RetrievalOverride(query=query, top_k=top_k),
            note="relaxed the query and widened the candidate set",
        )

    if move is RepairMove.NARROW:
        query = (rewritten_query or "").strip() or _narrow(working.query, missing)
        top_k = min(working.top_k or NARROW_TOP_K, NARROW_TOP_K)
        return RepairOutcome(
            move=move,
            working=RetrievalOverride(query=query, top_k=top_k),
            note="added the missing constraint to the query",
        )

    if move is RepairMove.SWITCH_STRATEGY:
        previous = sub_question.tool
        sub_question.tool = _PAIRED_TOOLS.get(previous, "semantic_search")
        return RepairOutcome(
            move=move,
            working=working,
            note=f"switched from {previous} to {sub_question.tool}",
        )

    if move is RepairMove.DECOMPOSE:
        parts = _split((rewritten_query or "").strip() or working.query)
        if len(parts) < 2:
            return RepairOutcome(
                move=move,
                working=working,
                note="the sub-question could not be split further",
            )
        children = [SubQuestion(text=part, tool=sub_question.tool) for part in parts]
        sub_question.abandon(f"decomposed into {len(parts)} sub-questions")
        return RepairOutcome(
            move=move,
            working=working,
            new_sub_questions=children,
            abandoned=True,
            note=f"split into {len(parts)} sub-questions",
        )

    document_id = _best_document(evidence)
    if document_id is None:
        return RepairOutcome(
            move=move,
            working=working,
            note="no retrieved chunk identified a document to fetch",
        )
    sub_question.tool = "fetch_document"
    return RepairOutcome(
        move=move,
        # The fetch_document tool takes the document id as its query, which is
        # why the id is both the query and a field: the field is what a trace
        # reader and the counters use, the query is what the tool receives.
        working=RetrievalOverride(
            query=str(document_id), top_k=working.top_k, document_id=document_id
        ),
        note=f"fetching document {document_id} whole",
    )


def _abandon_reason(sub_question: SubQuestion, failure: str) -> str:
    tried = [attempt.move.value for attempt in sub_question.attempts]
    if not tried:
        return f"abandoned without a repair being available: {failure}"
    return f"abandoned after trying {', '.join(tried)}: {failure}"


def _broaden(query: str) -> str:
    """Drop the qualifiers and the trailing restriction, in that order.

    Both edits shorten the query towards its subject, which is what a broader
    retrieval needs. If neither applies, the content words alone are used: a
    query that cannot be relaxed any other way is still better off without the
    grammar the store does not rank on.
    """
    words = [word for word in query.split() if word.lower().strip(",") not in _QUALIFIERS]
    relaxed = " ".join(words)
    lowered = relaxed.lower()
    cuts = [lowered.find(preposition) for preposition in _TRAILING_PREPOSITIONS]
    cut = min((position for position in cuts if position > 0), default=-1)
    if cut > 0:
        relaxed = relaxed[:cut].strip()
    if relaxed and relaxed != query:
        return relaxed
    content = [word for word in query.split() if word.lower() not in _STOPWORDS]
    return " ".join(content) if content else query


def _narrow(query: str, missing: str) -> str:
    missing = missing.strip()
    if not missing or missing.lower() in query.lower():
        return query
    return f"{query} {missing}"


def _split(text: str) -> list[str]:
    for splitter in _SPLITTERS:
        if splitter in text.lower():
            parts = _apply_split(text, splitter)
            if len(parts) > 1:
                return parts
    return [text.strip()]


def _apply_split(text: str, splitter: str) -> list[str]:
    lowered = text.lower()
    parts: list[str] = []
    start = 0
    while True:
        position = lowered.find(splitter, start)
        if position < 0:
            parts.append(text[start:].strip())
            break
        parts.append(text[start:position].strip())
        start = position + len(splitter)
    parts = [part for part in parts if part]
    if len(parts) < 2 or any(len(part.split()) < _MIN_PART_WORDS for part in parts):
        return [text.strip()]
    return parts


def _best_document(evidence: Sequence[RetrievedChunk]) -> int | None:
    if not evidence:
        return None
    best = max(evidence, key=lambda chunk: chunk.score if chunk.score is not None else 0.0)
    return best.document_id
