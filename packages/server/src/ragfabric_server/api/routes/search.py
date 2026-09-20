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

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ragfabric_core.auth.principal import AccessFilter, Principal
from ragfabric_core.db.session import get_db
from ragfabric_core.generate.answer import build_answer
from ragfabric_core.generate.cited import CitedAnswer, generate_cited_answer
from ragfabric_core.models.access import AuditLog
from ragfabric_core.models.document import QueryLog
from ragfabric_core.models.runs import RetrievalRun, Source
from ragfabric_core.providers.base import LLMProvider
from ragfabric_core.runtime import get_config
from ragfabric_core.store.vector_store import get_store
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
)

router = APIRouter()


def _context(payload: SearchRequest, principal: Principal, access: AccessFilter) -> RetrievalContext:
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
    rerank: str | None, registry: StrategyRegistry, llm: LLMProvider | None
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
    base = registry.get(StrategyName.TRADITIONAL)
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
    strategy = _strategy_for(payload.rerank, registry, llm)
    with start_trace() as tracing:
        result = strategy.retrieve(payload.query, _context(payload, principal, access))
        retrieval_ms = int((time.perf_counter() - started) * 1000)
        with trace("answer"):
            cited = generate_cited_answer(
                payload.query, result.chunks, llm, model=None, max_tokens=800
            )
        retrieved = [_chunk_to_row(c) for c in result.chunks]
        result_payload = build_answer(payload.query, retrieved, answer_text=cited.text)
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
    # Task 11 moves access_stats onto the vector stores and retires this
    # legacy in-memory index; until then it is the one place that number
    # lives, so it stays here rather than being duplicated early.
    before, after = get_store().access_stats(
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
        requested_strategy="traditional",
        selected_strategy="traditional",
        answer=result_payload["answer"],
        latency_ms=total_ms,
        retrieval_latency_ms=retrieval_ms,
        generation_latency_ms=total_ms - retrieval_ms,
        llm_calls=result.llm_calls + _cited_llm_calls(cited),
        retrieval_calls=result.retrieval_calls,
        input_tokens=result.input_tokens + cited.input_tokens,
        output_tokens=result.output_tokens + cited.output_tokens,
        estimated_cost_usd=0.0,
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
            strategy="traditional",
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
    return AnswerResponse(**result_payload)


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
    before, after = get_store().access_stats(
        {
            "collection_id": payload.collection_id,
            "document_id": payload.document_id,
            "format": payload.format,
        },
        access,
    )
    strategy = _strategy_for(payload.rerank, registry, llm)
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
    return SearchResults(query=payload.query, mode="semantic", results=results)


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
    before, after = get_store().access_stats(
        {
            "collection_id": payload.collection_id,
            "document_id": payload.document_id,
            "format": payload.format,
        },
        access,
    )
    strategy = _strategy_for(payload.rerank, registry, llm)
    vector_result = strategy.retrieve(payload.query, _context(payload, principal, access))
    filters = _lexical_filters(payload)
    lexical_hits = lexical.search(
        payload.query, payload.top_k * 3, access, filters=filters or None
    )
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
    return SearchResults(query=payload.query, mode="hybrid", results=results)


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
