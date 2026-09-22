"""POST /api/ask: retrieve, stream the answer, then record the run.

Event order is ``retrieval``, ``token``*, optionally ``superseded``,
``citations``, ``done``. Citations arrive last because a citation's ``used``
flag is derived from the markers the answer actually contains, which is
unknown until the final token: sending citations earlier would mean sending a
guess about which sources the answer leans on.

``superseded`` fires only when the streamed text fails the citation
contract. A streamed answer cannot be retried in place because the client
already rendered the rejected tokens, so the contract is still enforced but
the repair is announced: the corrected text and the reason are sent, and the
corrected text (not the streamed one) is what gets recorded. Recording an
answer that disagrees with what the user watched stream would be worse than
failing or announcing. ``docs/traditional-rag.md`` documents this event and
the SDK (``ragfabric_sdk``) handles it.

The run row is written after the stream finishes, never before: latency,
token counts and the citation list are only final once the last token has
been accounted for, and a row written early would leave ``/api/runs/{id}``
returning a half recorded run for as long as the client is still reading. The
``done`` event carries the ``run_id`` so a client can fetch the complete
record immediately.

The database session is deliberately NOT resolved through a FastAPI
dependency held open for the request. Retrieval (via the strategy's own
store, which opens and closes its own session per call) and generation both
run before any database session for this route exists at all; a short lived
session is opened, used and closed by ``_record`` only once streaming is
completely finished. Nothing in this module holds a session, or a pooled
connection, for as long as a client is reading the stream.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from ragfabric_core.auth.principal import AccessFilter, Principal
from ragfabric_core.db.session import SessionLocal
from ragfabric_core.generate.answer import build_answer
from ragfabric_core.generate.cited import (
    NO_EVIDENCE_ANSWER,
    SYSTEM_PROMPT,
    apply_agentic_contract,
    build_prompt,
    generate_cited_answer,
    sub_question_evidence_from,
)
from ragfabric_core.generate.contract import CitationViolation, assert_citation_contract
from ragfabric_core.models.access import AuditLog
from ragfabric_core.models.document import QueryLog
from ragfabric_core.models.runs import RetrievalRun, Source
from ragfabric_core.providers.base import LLMProvider, Message
from ragfabric_core.strategies.base import (
    RetrievalContext,
    RetrievedChunk,
    StrategyParams,
    StrategyRegistry,
    TraceSpan,
)
from ragfabric_core.telemetry.tracing import start_trace, trace
from ragfabric_server.api.routes.search import (
    _access_stats,
    _agent_fields,
    _chunk_to_row,
    _cited_llm_calls,
    _generate,
    _strategy_for,
    _usage,
)
from ragfabric_server.deps import (
    get_access_filter,
    get_embedding_model,
    get_llm_provider,
    get_principal,
    get_strategy_registry,
)
from ragfabric_server.schemas.ask import AskRequest
from ragfabric_server.schemas.search import AnswerResponse

router = APIRouter()


def _event(name: str, data: dict) -> str:
    return f"event: {name}\ndata: {json.dumps(data)}\n\n"


def _context(payload: AskRequest, principal: Principal, access: AccessFilter) -> RetrievalContext:
    """Build the strategy's per-request context.

    Kept as its own function rather than reused from ``search.py``: the two
    payload types are structurally identical for this purpose since Task 12
    (both carry ``top_k``, ``similarity_threshold`` and ``rerank``), but they
    remain distinct pydantic models with their own validation, and this
    module already keeps its own thin route layer independent of
    ``search.py`` except for the pieces it explicitly imports.
    """
    filters: dict[str, str | int | float | bool] = {}
    if payload.document_id is not None:
        filters["document_id"] = payload.document_id
    if payload.format is not None:
        filters["format"] = payload.format
    return RetrievalContext(
        principal=principal,
        access_filter=access,
        collection_ids=[payload.collection_id] if payload.collection_id is not None else None,
        params=StrategyParams(
            top_k=payload.top_k,
            similarity_threshold=payload.similarity_threshold,
            metadata_filters=filters,
        ),
    )


def _rows(chunks: list[RetrievedChunk]) -> list[dict]:
    """Reuse ``search.py``'s per-chunk dict mapping rather than restating it."""
    return [_chunk_to_row(c) for c in chunks]


def _record(
    *,
    payload: AskRequest,
    principal: Principal,
    access: AccessFilter,
    strategy,
    result,
    answer: dict,
    spans: list[dict],
    embedding_model: str,
    llm_calls: int,
    input_tokens: int,
    output_tokens: int,
    total_ms: int,
    retrieval_ms: int,
) -> int:
    """Open one short lived session, write every row for this run, and return its id.

    Called exactly once, strictly after the stream (or the non-streaming
    generation) has finished, so this is the only point at which the route
    touches the database at all.

    ``access_stats`` (candidate counts before/after the access filter, for the
    audit row's ``sources_filtered``) is read through ``_access_stats`` off
    the exact strategy that just retrieved, so the count is measured against
    the same candidates this request actually searched, the same as
    ``search.py``'s three endpoints, whichever store that strategy uses.
    """
    used = {c["chunk_id"] for c in answer["citations"] if c["used"]}
    before, after = _access_stats(
        strategy,
        {
            "collection_id": payload.collection_id,
            "document_id": payload.document_id,
            "format": payload.format,
        },
        access,
    )
    with SessionLocal() as db:
        run = RetrievalRun(
            user_id=principal.user_id,
            api_key_id=principal.api_key_id,
            question=payload.query,
            mode="manual",
            requested_strategy=payload.strategy,
            selected_strategy=payload.strategy,
            answer=answer["answer"],
            latency_ms=total_ms,
            retrieval_latency_ms=retrieval_ms,
            generation_latency_ms=max(total_ms - retrieval_ms, 0),
            llm_calls=llm_calls,
            retrieval_calls=result.retrieval_calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=None,
            embedding_model=embedding_model,
            trace=spans,
        )
        db.add(run)
        db.flush()
        for rank, row in enumerate(_rows(result.chunks), start=1):
            db.add(
                Source(
                    retrieval_run_id=run.id,
                    chunk_id=row["chunk_id"],
                    document_id=row["document_id"],
                    rank=rank,
                    score=row["score"],
                    cited=row["chunk_id"] in used,
                    page=row["page"],
                )
            )
        db.add(
            AuditLog(
                principal_user_id=principal.user_id,
                api_key_id=principal.api_key_id,
                action="query",
                question=payload.query,
                strategy=payload.strategy,
                retrieval_run_id=run.id,
                sources_returned=len(result.chunks),
                sources_filtered=max(before - after, 0),
                details={"collection_id": payload.collection_id, "endpoint": "ask"},
            )
        )
        db.add(
            QueryLog(
                user_id=principal.user_id,
                collection_id=payload.collection_id,
                question=payload.query,
                confidence=answer["confidence"],
                cited_document_ids=sorted(
                    {
                        c["document_id"]
                        for c in answer["citations"]
                        if c["used"] and c["document_id"] is not None
                    }
                ),
            )
        )
        db.commit()
        return run.id


@router.post("/ask", response_model=None)
def ask(
    payload: AskRequest,
    principal: Principal = Depends(get_principal),
    access: AccessFilter = Depends(get_access_filter),
    registry: StrategyRegistry = Depends(get_strategy_registry),
    llm: LLMProvider = Depends(get_llm_provider),
    embedding_model: str = Depends(get_embedding_model),
) -> StreamingResponse | AnswerResponse:
    # AskRequest.strategy is pattern-validated to a name the registry holds,
    # so an unknown one is a 422 from validation and never a KeyError here.
    # _strategy_for additionally honours payload.rerank, building a
    # per-request strategy around the shared store/embedder rather than
    # mutating the shared one when a reranker override is present (the
    # rerank override applies to the traditional strategy only).
    strategy = _strategy_for(payload.rerank, registry, llm, payload.strategy)

    if not payload.stream:
        started = time.perf_counter()
        with start_trace() as tracing:
            result = strategy.retrieve(payload.query, _context(payload, principal, access))
            retrieval_ms = int((time.perf_counter() - started) * 1000)
            with trace("answer"):
                generated = _generate(payload.query, result, llm)
            answer = build_answer(payload.query, _rows(result.chunks), answer_text=generated.text)
        total_ms = int((time.perf_counter() - started) * 1000)
        _record(
            payload=payload,
            principal=principal,
            access=access,
            strategy=strategy,
            result=result,
            answer=answer,
            spans=[s.model_dump() for s in result.trace] + [s.model_dump() for s in tracing.spans],
            embedding_model=embedding_model,
            llm_calls=result.llm_calls + generated.llm_calls,
            input_tokens=result.input_tokens + generated.input_tokens,
            output_tokens=result.output_tokens + generated.output_tokens,
            total_ms=total_ms,
            retrieval_ms=retrieval_ms,
        )
        return AnswerResponse(
            **answer, usage=_usage(result, generated), **_agent_fields(result, generated)
        )

    def events() -> Iterator[str]:
        # Deliberately NOT wrapped in ``start_trace()``/``trace()``: those use
        # a contextvars.ContextVar whose token is set on __enter__ and reset
        # on __exit__, which requires both to run in the same context.
        # Starlette iterates a sync generator response one `next()` call per
        # dispatch to a worker thread, and a `with` block held open across a
        # `yield` that crosses that boundary is entered in one context and
        # exited in another, which raises
        # ``ValueError: ... was created in a different Context`` (reproduced
        # while building this route). Retrieval's own spans are unaffected,
        # since ``strategy.retrieve()`` collects ``result.trace`` in a plain
        # local list, not through this contextvar. The generation phase's
        # span is instead timed by hand below and OTel export for that one
        # phase is not attempted on the streaming path; the non-streaming
        # branch above runs start to finish in a single call and keeps full
        # parity with ``/api/search/query``.
        started = time.perf_counter()
        result = strategy.retrieve(payload.query, _context(payload, principal, access))
        retrieval_ms = int((time.perf_counter() - started) * 1000)
        yield _event(
            "retrieval",
            {
                "chunks": len(result.chunks),
                "strategy": str(result.strategy),
                "trace": [s.model_dump() for s in result.trace],
                # Empty for every strategy that does not decompose. Sent on
                # this event rather than at the end because it is known the
                # moment retrieval finishes, and a client rendering progress
                # can say which part of the question is already answered while
                # the tokens are still arriving.
                "sub_questions": [report.model_dump() for report in result.sub_questions],
            },
        )

        generation_started = time.perf_counter()
        if not result.chunks:
            # Mirrors generate_cited_answer's own no-chunks short circuit: no
            # model call is made, so this branch adds nothing to
            # result.llm_calls/tokens, never a guess at what a call would
            # have cost.
            text = NO_EVIDENCE_ANSWER
            yield _event("token", {"text": text})
            llm_calls = result.llm_calls
            in_tokens = out_tokens = 0
        else:
            pieces: list[str] = []
            for delta in llm.stream(
                [
                    Message(role="system", content=SYSTEM_PROMPT),
                    Message(role="user", content=build_prompt(payload.query, result.chunks)),
                ],
                max_tokens=800,
            ):
                pieces.append(delta)
                yield _event("token", {"text": delta})
            text = "".join(pieces)
            # The stream protocol yields text only, no usage, so an accepted
            # streamed answer honestly reports zero tokens here rather than a
            # number the provider never gave us.
            in_tokens = out_tokens = 0
            try:
                assert_citation_contract(text, result.chunks)
            except CitationViolation:
                # Streaming cannot be retried in place: the client already
                # rendered the rejected tokens. Fall back for the RECORDED
                # answer and announce the repair rather than silently
                # recording an answer the client never saw.
                if result.sub_questions:
                    # The agent's repair is to remove what the contract
                    # refused, claim by claim, which costs no further call:
                    # the streamed text is edited, not regenerated. Falling
                    # back to a second generation here would spend a call to
                    # rewrite the parts that were already sound.
                    repaired = apply_agentic_contract(
                        text,
                        result.chunks,
                        sub_question_evidence_from(result.sub_questions, result.chunks),
                    )
                    text = repaired.text or NO_EVIDENCE_ANSWER
                    llm_calls = result.llm_calls + 1
                    yield _event(
                        "superseded", {"text": text, "reason": "unsupported claims removed"}
                    )
                else:
                    cited = generate_cited_answer(payload.query, result.chunks, llm)
                    text = cited.text
                    in_tokens, out_tokens = cited.input_tokens, cited.output_tokens
                    # One real call was already spent on the rejected stream;
                    # _cited_llm_calls (search.py, reused rather than
                    # reimplemented) accounts correctly for however many more
                    # calls the fallback itself made, including its own retry.
                    llm_calls = result.llm_calls + 1 + _cited_llm_calls(cited)
                    yield _event("superseded", {"text": text, "reason": "citation contract"})
            else:
                # Exactly one real call was made and it was accepted.
                llm_calls = result.llm_calls + 1
        generation_span = TraceSpan(
            name="answer_stream",
            started_ms=int((generation_started - started) * 1000),
            duration_ms=int((time.perf_counter() - generation_started) * 1000),
        )

        answer = build_answer(payload.query, _rows(result.chunks), answer_text=text)
        yield _event("citations", {"citations": answer["citations"]})
        total_ms = int((time.perf_counter() - started) * 1000)
        spans = [s.model_dump() for s in result.trace] + [generation_span.model_dump()]

        run_id = _record(
            payload=payload,
            principal=principal,
            access=access,
            strategy=strategy,
            result=result,
            answer=answer,
            spans=spans,
            embedding_model=embedding_model,
            llm_calls=llm_calls,
            input_tokens=result.input_tokens + in_tokens,
            output_tokens=result.output_tokens + out_tokens,
            total_ms=total_ms,
            retrieval_ms=retrieval_ms,
        )
        yield _event(
            "done",
            {
                "run_id": run_id,
                "latency_ms": total_ms,
                # Assembled by hand rather than through _usage: on the
                # streaming path llm_calls and the token counts were tracked
                # above across the accepted/superseded branches, and _usage
                # would recompute them from a CitedAnswer this branch may not
                # have. Every number here is still one that was counted.
                "usage": {
                    "embedding_calls": result.embedding_calls,
                    "llm_calls": llm_calls,
                    "retrieval_calls": result.retrieval_calls,
                    "input_tokens": result.input_tokens + in_tokens,
                    "output_tokens": result.output_tokens + out_tokens,
                },
            },
        )

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
