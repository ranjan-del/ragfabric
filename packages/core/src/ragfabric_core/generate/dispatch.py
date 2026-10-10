"""Choose and run the generator a retrieval result deserves.

Moved here from the server's ``search.py`` in Phase 8 (design decision D5) so
the API and the evaluation harness answer through one implementation. An
evaluation of a different code path than the one users get would measure
nothing. The server re-exports these names, so its routes and tests are
unchanged.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ragfabric_core.generate.cited import (
    CitedAnswer,
    DatedSubQuestion,
    DroppedClaim,
    dated_sources_note,
    dated_sub_questions,
    generate_agentic_answer,
    generate_cited_answer,
    generate_graph_answer,
    sub_question_evidence_from,
)
from ragfabric_core.providers.base import LLMProvider
from ragfabric_core.strategies.base import RetrievalResult, StrategyName


class Generated(BaseModel):
    """One generation step, in the form every route accounts for it.

    Both generators (the single shot one from Phase 3 and the agent's, which
    drops what the contract refuses instead of retrying) report through this,
    so the routes below never branch on which one ran except where the answer
    genuinely differs.
    """

    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    # Real calls issued by this step, counted rather than inferred from the
    # generator that produced the text (ADR 0004).
    llm_calls: int = 0
    dropped_claims: list[DroppedClaim] = Field(default_factory=list)
    dated_sources: list[DatedSubQuestion] = Field(default_factory=list)
    # Graph path only: claims the graph citation contract refused.
    dropped_relationship_claims: list[DroppedClaim] = Field(default_factory=list)


def uses_graph_path(result: RetrievalResult) -> bool:
    """A result generated against its walk: the graph strategy's, or an agent's that walked one."""
    return result.strategy == StrategyName.GRAPH or (
        result.subgraph is not None and bool(result.subgraph.edges)
    )


def generate_for_result(query: str, result, llm: LLMProvider) -> Generated:
    """Generate the answer this result deserves.

    A result carrying sub-question reports came from the agent, and the agent's
    answer is checked claim by claim: an unsupported claim is removed and the
    removal reported, rather than the whole answer being regenerated. There is
    nothing to regenerate for, because a claim the evidence does not support is
    the model saying more than it was given.
    """
    if uses_graph_path(result):
        # Generated against the chunks and the walked sub-graph together, and
        # checked claim by claim: a relationship claim must cite a traversed
        # edge and a passage that backs it (graph citation contract, rulings
        # R32 and R33), a chunk claim the Phase 3 contract. One call, no retry.
        graph = generate_graph_answer(query, result.chunks, result.subgraph, llm)
        text = graph.text
        reports: list[DatedSubQuestion] = []
        if result.sub_questions:
            # An agent that walked a graph keeps what the agentic path gives it:
            # sub-questions whose sources carry differing effective dates are
            # reported, and the answer says so in the same words.
            reports = dated_sub_questions(
                sub_question_evidence_from(result.sub_questions, result.chunks), result.chunks
            )
            note = dated_sources_note(reports)
            if note:
                text = f"{text} {note}"
        return Generated(
            text=text,
            dated_sources=reports,
            input_tokens=graph.input_tokens,
            output_tokens=graph.output_tokens,
            # Exactly one call, or none at all when there were no chunks.
            llm_calls=1 if graph.generator == "llm" else 0,
            dropped_claims=graph.dropped_claims,
            dropped_relationship_claims=graph.dropped_relationship_claims,
        )
    if result.sub_questions:
        agentic = generate_agentic_answer(
            query,
            result.chunks,
            llm,
            sub_question_evidence=sub_question_evidence_from(result.sub_questions, result.chunks),
        )
        return Generated(
            text=agentic.text,
            input_tokens=agentic.input_tokens,
            output_tokens=agentic.output_tokens,
            # Exactly one call, or none at all on the empty-pool path.
            llm_calls=1 if agentic.generator == "llm" else 0,
            dropped_claims=agentic.dropped_claims,
            dated_sources=agentic.dated_sources,
        )
    cited = generate_cited_answer(query, result.chunks, llm, model=None, max_tokens=800)
    return Generated(
        text=cited.text,
        input_tokens=cited.input_tokens,
        output_tokens=cited.output_tokens,
        llm_calls=cited_llm_calls(cited),
    )


def cited_llm_calls(cited: CitedAnswer) -> int:
    """How many real model calls ``generate_cited_answer`` actually issued.

    ``generator == "extractive"`` does NOT mean zero calls: on the fallback
    path (two rejected attempts) two real calls were made before the
    extractive generator produced the text. The only case with zero calls is
    the no-chunks short-circuit, which returns ``generator == "extractive"``
    and ``retried == False``. So:

      - ``retried`` is only ever set True after a genuine second attempt, so
        it always means exactly two calls were issued, regardless of which
        generator ended up producing the text.
      - not retried and ``generator == "llm"`` means the first attempt was
        accepted: exactly one call.
      - not retried and ``generator == "extractive"`` is the no-chunks path:
        no call was made at all.
    """
    if cited.retried:
        return 2
    if cited.generator == "extractive":
        return 0
    return 1
