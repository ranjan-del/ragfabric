"""Search + answer routes.

- ``POST /query`` runs retrieval (semantic or hybrid) then assembles a cited
  answer with confidence, citations, highlighted supporting text, and the source
  document, and logs the question for analytics. It also records a
  ``RetrievalRun`` with its ``Source`` rows and an ``AuditLog`` row, so every
  answer can be traced and audited later.
- ``POST /semantic`` and ``POST /hybrid`` return raw ranked chunks for callers
  that want to build their own UI over the results, and each writes an
  ``AuditLog`` row too.
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ragfabric_core.auth.principal import AccessFilter, Principal
from ragfabric_core.db.session import get_db
from ragfabric_core.generate.answer import build_answer
from ragfabric_core.ingest.embed import get_embedder
from ragfabric_core.models.access import AuditLog
from ragfabric_core.models.document import QueryLog
from ragfabric_core.models.runs import RetrievalRun, Source
from ragfabric_core.retrieve.hybrid import HybridRetriever
from ragfabric_core.retrieve.retriever import Retriever
from ragfabric_core.store.vector_store import get_store
from ragfabric_core.telemetry.tracing import start_trace, trace
from ragfabric_server.deps import get_access_filter, get_principal
from ragfabric_server.schemas.search import AnswerResponse, SearchRequest, SearchResults

router = APIRouter()


def _retrieve(payload: SearchRequest, access: AccessFilter) -> list[dict]:
    """Dispatch to the semantic or hybrid retriever based on the request mode."""
    if payload.mode == "hybrid":
        retriever = HybridRetriever()
    else:
        retriever = Retriever()
    return retriever.retrieve(
        payload.query,
        top_k=payload.top_k,
        collection_id=payload.collection_id,
        document_id=payload.document_id,
        format=payload.format,
        access=access,
    )


@router.post("/query", response_model=AnswerResponse)
def query(
    payload: SearchRequest,
    db: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
    access: AccessFilter = Depends(get_access_filter),
) -> AnswerResponse:
    """Ask a question and get a cited, grounded answer."""
    started = time.perf_counter()
    with start_trace() as ctx:
        retrieved = _retrieve(payload, access)
        retrieval_ms = int((time.perf_counter() - started) * 1000)
        with trace("answer"):
            result = build_answer(payload.query, retrieved)
    total_ms = int((time.perf_counter() - started) * 1000)

    # Record only the documents the answer actually cited, so the analytics
    # "most referenced" panel reflects usage rather than corpus size.
    cited = sorted(
        {
            c["document_id"]
            for c in result["citations"]
            if c["used"] and c["document_id"] is not None
        }
    )
    used_chunks = {c["chunk_id"] for c in result["citations"] if c["used"]}
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
        answer=result["answer"],
        latency_ms=total_ms,
        retrieval_latency_ms=retrieval_ms,
        generation_latency_ms=total_ms - retrieval_ms,
        llm_calls=0,
        retrieval_calls=1,
        input_tokens=0,
        output_tokens=0,
        estimated_cost_usd=0.0,
        embedding_model=f"hashing-{get_embedder().dim}",
        trace=[s.model_dump() for s in ctx.spans],
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
            confidence=result["confidence"],
            cited_document_ids=cited,
        )
    )
    db.commit()
    return AnswerResponse(**result)


@router.post("/semantic", response_model=SearchResults)
def semantic_search(
    payload: SearchRequest,
    db: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
    access: AccessFilter = Depends(get_access_filter),
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
    results = _retrieve(payload, access)
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
) -> SearchResults:
    """Return chunks ranked by a blend of lexical and semantic scores."""
    payload.mode = "hybrid"
    before, after = get_store().access_stats(
        {
            "collection_id": payload.collection_id,
            "document_id": payload.document_id,
            "format": payload.format,
        },
        access,
    )
    results = _retrieve(payload, access)
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
