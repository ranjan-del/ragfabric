"""Search + answer routes.

- ``POST /query`` runs retrieval through the traditional RAG strategy, generates
  a cited answer through the configured LLM provider (falling back to the
  extractive generator on a citation contract violation), then assembles a
  response with confidence, citations, highlighted supporting text, and the
  source document, and logs the question for analytics. It also records a
  ``RetrievalRun`` with its ``Source`` rows and an ``AuditLog`` row, so every
  answer can be traced and audited later.
- ``POST /semantic`` and ``POST /hybrid`` return raw ranked chunks for callers
  that want to build their own UI over the results, and each writes an
  ``AuditLog`` row too.

Task 10 moves all three endpoints off the v1 in-memory index and onto the real
stores built in Tasks 2-9: the traditional strategy for the vector side, and
``PostgresLexicalStore`` for the keyword side of ``/hybrid``.
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ragfabric_core.auth.principal import AccessFilter, Principal
from ragfabric_core.db.session import get_db
from ragfabric_core.generate.answer import build_answer
from ragfabric_core.generate.cited import (
    CitedAnswer,
    DatedSubQuestion,
    DroppedClaim,
    generate_agentic_answer,
    generate_cited_answer,
    sub_question_evidence_from,
)
from ragfabric_core.models.access import AuditLog
from ragfabric_core.models.document import QueryLog
from ragfabric_core.models.runs import RetrievalRun, Source
from ragfabric_core.providers.base import LLMProvider
from ragfabric_core.runtime import get_config
from ragfabric_core.stores.base import LexicalStore
from ragfabric_core.strategies.base import (
    RetrievalContext,
    RetrievedChunk,
    RetrieverStrategy,
    StrategyName,
    StrategyParams,
    StrategyRegistry,
)
from ragfabric_core.strategies.traditional import TraditionalRAGStrategy
from ragfabric_core.telemetry.tracing import start_trace, trace
from ragfabric_server.deps import (
    get_access_filter,
    get_embedding_model,
    get_lexical_store,
    get_llm_provider,
    get_principal,
    get_reranker,
    get_strategy_registry,
)
from ragfabric_server.schemas.search import (
    AnswerResponse,
    SearchRequest,
    SearchResultItem,
    SearchResults,
    Usage,
)

router = APIRouter()


def _context(
    payload: SearchRequest, principal: Principal, access: AccessFilter
) -> RetrievalContext:
    """Build the strategy's per-request context.

    ``similarity_threshold`` (Task 12) is threaded straight through to
    ``StrategyParams``, whose own default of ``0.0`` reproduces the prior
    behaviour (no floor) whenever a caller leaves the field unset.
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


def _configured_max_context_tokens(cfg) -> int:
    """Mirrors ``registry_defaults.default_registry``'s own coercion, so a per
    request strategy built around a different reranker keeps the exact same
    budget the shared strategy was built with."""
    value = cfg.strategies.traditional.get("max_context_tokens")
    try:
        return int(value)
    except (TypeError, ValueError):
        return 6000


def _strategy_for(
    rerank: str | None,
    registry: StrategyRegistry,
    llm: LLMProvider | None,
    name: str = StrategyName.TRADITIONAL,
) -> RetrieverStrategy:
    """Return the strategy this request should retrieve through.

    ``rerank is None`` (the default, absent from the request body) returns the
    registry's shared ``TraditionalRAGStrategy`` instance unchanged: that
    instance is a process-wide singleton (``get_strategy_registry``) serving
    every concurrent request, so it must never be mutated to apply one
    caller's choice.

    Naming a reranker, including ``"none"``, builds a FRESH
    ``TraditionalRAGStrategy`` around the SAME shared vector store and
    embedding provider (read off the shared strategy's public ``store`` and
    ``embedder`` properties, never a private attribute) with only the
    reranker swapped in. The fresh instance is local to this request/response
    cycle and is discarded afterwards, so one caller's reranker choice can
    never be observed by another in-flight request.

    The STRATEGY WRAPPER is what gets rebuilt per request, and that part is
    genuinely cheap: it is a thin object holding references, not a load. The
    reranker itself is NOT rebuilt per request: it comes from
    ``deps.get_reranker``, a process-wide cache keyed by kind. Without that
    cache, naming ``rerank: "cross_encoder"`` would reconstruct
    ``CrossEncoderReranker()`` (and reload its model from disk) on every
    single request; the cache makes repeated cross-encoder requests as cheap
    as the ``none``/``llm`` cases always were.
    """
    base = registry.get(StrategyName(name))
    if rerank is None:
        return base
    if not isinstance(base, TraditionalRAGStrategy):
        # Only the traditional strategy is rerank-overridable in this phase;
        # any other registered strategy is returned untouched.
        return base
    reranker = get_reranker(rerank, llm=llm if rerank == "llm" else None)
    cfg = get_config()
    return TraditionalRAGStrategy(
        embedding_provider=base.embedder,
        vector_store=base.store,
        reranker=reranker,
        max_context_tokens=_configured_max_context_tokens(cfg),
        generation_model=cfg.llm.model,
    )


def _access_stats(
    strategy: RetrieverStrategy, filters: dict, access: AccessFilter
) -> tuple[int, int]:
    """Candidate counts before and after the access filter, for the audit row.

    Read off whichever store the strategy actually searched: the vector store
    for the traditional strategy, the BM25 store for the vectorless one. Both
    expose access_stats, so the number stays a measurement on either path
    rather than a zero standing in for "not looked at" (ADR 0004).
    """
    store = _countable_store(strategy)
    if store is None:
        raise RuntimeError(f"{type(strategy).__name__} exposes no store to count candidates on")
    return store.access_stats(filters, access)


def _countable_store(strategy: RetrieverStrategy):
    """The store the audit row's candidate counts are measured on.

    The traditional strategy searches a vector store and the vectorless one a
    BM25 store, so each exposes the store it queried. The agent has neither:
    it holds tools, each wrapping one of those strategies, and it may search
    through more than one of them in a single run. Counting the union would
    mean adding candidate counts from two different indexes over the same
    corpus, which is not a number that means anything, so the count is taken on
    the first search tool the agent was built with and is a measurement on that
    index. Summing them would look more thorough and be less true (ADR 0004).
    """
    for attribute in ("store", "bm25_store"):
        store = getattr(strategy, attribute, None)
        if store is not None and hasattr(store, "access_stats"):
            return store
    for tool in (getattr(strategy, "tools", None) or {}).values():
        inner = getattr(tool, "strategy", None)
        if inner is None:
            continue
        store = _countable_store(inner)
        if store is not None:
            return store
    return None


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


def _generate(query: str, result, llm: LLMProvider) -> Generated:
    """Generate the answer this result deserves.

    A result carrying sub-question reports came from the agent, and the agent's
    answer is checked claim by claim: an unsupported claim is removed and the
    removal reported, rather than the whole answer being regenerated. There is
    nothing to regenerate for, because a claim the evidence does not support is
    the model saying more than it was given.
    """
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
        llm_calls=_cited_llm_calls(cited),
    )


def _agent_fields(result, generated: Generated | None = None) -> dict:
    """The response fields only an agent fills in, empty for everything else."""
    return {
        "sub_questions": [report.model_dump() for report in result.sub_questions],
        "dropped_claims": [claim.model_dump() for claim in generated.dropped_claims]
        if generated is not None
        else [],
        "dated_sources": [dated.model_dump() for dated in generated.dated_sources]
        if generated is not None
        else [],
        "trace": [span.model_dump() for span in result.trace] if result.sub_questions else [],
    }


def _usage(result, generated: Generated | None = None) -> Usage:
    """Assemble the response's usage block out of counts, never estimates."""
    return Usage(
        embedding_calls=result.embedding_calls,
        llm_calls=result.llm_calls + (generated.llm_calls if generated is not None else 0),
        retrieval_calls=result.retrieval_calls,
        input_tokens=result.input_tokens + (generated.input_tokens if generated is not None else 0),
        output_tokens=result.output_tokens
        + (generated.output_tokens if generated is not None else 0),
    )


def _lexical_filters(payload: SearchRequest) -> dict[str, str | int]:
    """The metadata filters ``/hybrid`` applies to both the vector and lexical legs."""
    filters: dict[str, str | int] = {}
    if payload.collection_id is not None:
        filters["collection_id"] = payload.collection_id
    if payload.document_id is not None:
        filters["document_id"] = payload.document_id
    if payload.format is not None:
        filters["format"] = payload.format
    return filters


def _to_result_item(
    chunk: RetrievedChunk,
    *,
    lexical_score: float | None = None,
    hybrid_score: float | None = None,
) -> SearchResultItem:
    return SearchResultItem(
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        filename=chunk.metadata.get("filename"),
        format=chunk.metadata.get("format"),
        page=chunk.page,
        # RetrievedChunk carries no chunk_index (only the legacy in-memory
        # store's dicts did); left unset rather than guessed at.
        chunk_index=None,
        score=chunk.score if chunk.score is not None else 0.0,
        lexical_score=lexical_score,
        hybrid_score=hybrid_score,
        text=chunk.text,
    )


def _chunk_to_row(chunk: RetrievedChunk) -> dict:
    """``build_answer``/``Source`` still work over plain dicts (unchanged by Task 10)."""
    return {
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "collection_id": chunk.collection_id,
        "text": chunk.text,
        "page": chunk.page,
        "score": chunk.score,
        "char_start": chunk.char_start,
        "char_end": chunk.char_end,
        "filename": chunk.metadata.get("filename"),
        "format": chunk.metadata.get("format"),
    }


def _cited_llm_calls(cited: CitedAnswer) -> int:
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


@router.post("/query", response_model=AnswerResponse)
def query(
    payload: SearchRequest,
    db: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
    access: AccessFilter = Depends(get_access_filter),
    registry: StrategyRegistry = Depends(get_strategy_registry),
    llm: LLMProvider = Depends(get_llm_provider),
    embedding_model: str = Depends(get_embedding_model),
) -> AnswerResponse:
    """Ask a question and get a cited, grounded answer."""
    started = time.perf_counter()
    strategy = _strategy_for(payload.rerank, registry, llm, payload.strategy)
    with start_trace() as tracing:
        result = strategy.retrieve(payload.query, _context(payload, principal, access))
        retrieval_ms = int((time.perf_counter() - started) * 1000)
        with trace("answer"):
            generated = _generate(payload.query, result, llm)
        retrieved = [_chunk_to_row(c) for c in result.chunks]
        result_payload = build_answer(payload.query, retrieved, answer_text=generated.text)
    total_ms = int((time.perf_counter() - started) * 1000)

    # Record only the documents the answer actually cited, so the analytics
    # "most referenced" panel reflects usage rather than corpus size.
    cited_document_ids = sorted(
        {
            c["document_id"]
            for c in result_payload["citations"]
            if c["used"] and c["document_id"] is not None
        }
    )
    used_chunks = {c["chunk_id"] for c in result_payload["citations"] if c["used"]}
    # access_stats lives on the vector store itself (Task 13), called on the
    # exact store instance that just answered this request, so the count is
    # measured against the same candidates the strategy actually searched.
    before, after = _access_stats(
        strategy,
        {
            "collection_id": payload.collection_id,
            "document_id": payload.document_id,
            "format": payload.format,
        },
        access,
    )
    run = RetrievalRun(
        user_id=principal.user_id,
        api_key_id=principal.api_key_id,
        question=payload.query,
        mode="manual",
        requested_strategy=payload.strategy,
        selected_strategy=payload.strategy,
        answer=result_payload["answer"],
        latency_ms=total_ms,
        retrieval_latency_ms=retrieval_ms,
        generation_latency_ms=total_ms - retrieval_ms,
        llm_calls=result.llm_calls + generated.llm_calls,
        retrieval_calls=result.retrieval_calls,
        input_tokens=result.input_tokens + generated.input_tokens,
        output_tokens=result.output_tokens + generated.output_tokens,
        estimated_cost_usd=None,
        embedding_model=embedding_model,
        trace=[s.model_dump() for s in result.trace] + [s.model_dump() for s in tracing.spans],
    )
    db.add(run)
    db.flush()
    for rank, row in enumerate(retrieved, start=1):
        db.add(
            Source(
                retrieval_run_id=run.id,
                chunk_id=row.get("chunk_id"),
                document_id=row.get("document_id"),
                rank=rank,
                score=row.get("score"),
                cited=row.get("chunk_id") in used_chunks,
                page=row.get("page"),
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
            sources_returned=len(retrieved),
            sources_filtered=max(before - after, 0),
            details={"collection_id": payload.collection_id, "mode": payload.mode},
        )
    )
    db.add(
        QueryLog(
            user_id=principal.user_id,
            collection_id=payload.collection_id,
            question=payload.query,
            confidence=result_payload["confidence"],
            cited_document_ids=cited_document_ids,
        )
    )
    db.commit()
    return AnswerResponse(
        **result_payload, usage=_usage(result, generated), **_agent_fields(result, generated)
    )


@router.post("/semantic", response_model=SearchResults)
def semantic_search(
    payload: SearchRequest,
    db: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
    access: AccessFilter = Depends(get_access_filter),
    registry: StrategyRegistry = Depends(get_strategy_registry),
    llm: LLMProvider = Depends(get_llm_provider),
) -> SearchResults:
    """Return the most semantically similar chunks for a query."""
    payload.mode = "semantic"
    strategy = _strategy_for(payload.rerank, registry, llm, payload.strategy)
    before, after = _access_stats(
        strategy,
        {
            "collection_id": payload.collection_id,
            "document_id": payload.document_id,
            "format": payload.format,
        },
        access,
    )
    result = strategy.retrieve(payload.query, _context(payload, principal, access))
    results = [_to_result_item(c) for c in result.chunks]
    db.add(
        AuditLog(
            principal_user_id=principal.user_id,
            api_key_id=principal.api_key_id,
            action="search",
            question=payload.query,
            strategy="semantic",
            sources_returned=len(results),
            sources_filtered=max(before - after, 0),
            details={"collection_id": payload.collection_id, "mode": payload.mode},
        )
    )
    db.commit()
    return SearchResults(
        query=payload.query, mode="semantic", strategy=payload.strategy, results=results
    )


@router.post("/hybrid", response_model=SearchResults)
def hybrid_search(
    payload: SearchRequest,
    db: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
    access: AccessFilter = Depends(get_access_filter),
    registry: StrategyRegistry = Depends(get_strategy_registry),
    lexical: LexicalStore = Depends(get_lexical_store),
    llm: LLMProvider = Depends(get_llm_provider),
) -> SearchResults:
    """Return chunks ranked by a blend of lexical and semantic scores.

    The fusion is a simple normalised score sum over the vector-side and
    lexical-side hits, the same approach the v1 hybrid retriever used. Phase 4
    replaces this with BM25 plus phrase and identifier boosting; doing that
    here would pull Phase 4's whole subject forward into this task.
    """
    payload.mode = "hybrid"
    if payload.strategy != StrategyName.TRADITIONAL:
        # Hybrid is defined as one vector ranking fused with one lexical
        # ranking. Substituting a lexical-only strategy for the vector leg
        # would fuse the lexical ranking with itself and report a
        # hybrid_score that means nothing. Refusing is honest; serving it and
        # calling the result hybrid is not. Use /api/search/semantic or
        # /api/search/query with strategy: vectorless instead.
        raise HTTPException(
            status_code=422,
            detail=(
                f"hybrid search fuses one vector ranking with one lexical ranking, so it "
                f"cannot run the {payload.strategy} strategy, which does not provide that "
                f"pair; use /api/search/semantic or /api/search/query instead"
            ),
        )
    strategy = _strategy_for(payload.rerank, registry, llm, payload.strategy)
    before, after = _access_stats(
        strategy,
        {
            "collection_id": payload.collection_id,
            "document_id": payload.document_id,
            "format": payload.format,
        },
        access,
    )
    vector_result = strategy.retrieve(payload.query, _context(payload, principal, access))
    filters = _lexical_filters(payload)
    lexical_hits = lexical.search(payload.query, payload.top_k * 3, access, filters=filters or None)
    fused = _fuse(vector_result.chunks, lexical_hits, payload.top_k)
    results = [
        _to_result_item(chunk, lexical_score=lex_score, hybrid_score=vec_score + lex_score)
        for chunk, vec_score, lex_score in fused
    ]
    db.add(
        AuditLog(
            principal_user_id=principal.user_id,
            api_key_id=principal.api_key_id,
            action="search",
            question=payload.query,
            strategy="hybrid",
            sources_returned=len(results),
            sources_filtered=max(before - after, 0),
            details={"collection_id": payload.collection_id, "mode": payload.mode},
        )
    )
    db.commit()
    return SearchResults(
        query=payload.query, mode="hybrid", strategy=payload.strategy, results=results
    )


def _fuse(
    vector_hits: list[RetrievedChunk], lexical_hits: list[RetrievedChunk], top_k: int
) -> list[tuple[RetrievedChunk, float, float]]:
    """Normalised score sum. Phase 4 replaces this with BM25 plus boosting."""

    def norm(hits: list[RetrievedChunk]) -> dict[int, float]:
        scores = [h.score or 0.0 for h in hits]
        top = max(scores, default=0.0)
        return {h.chunk_id: ((h.score or 0.0) / top if top > 0 else 0.0) for h in hits}

    v, lex = norm(vector_hits), norm(lexical_hits)
    by_id = {h.chunk_id: h for h in vector_hits} | {h.chunk_id: h for h in lexical_hits}
    fused = [(by_id[cid], v.get(cid, 0.0), lex.get(cid, 0.0)) for cid in by_id]
    fused.sort(key=lambda row: (-(row[1] + row[2]), row[0].chunk_id))
    return fused[:top_k]
