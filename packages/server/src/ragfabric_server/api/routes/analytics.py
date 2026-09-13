"""Analytics routes powering the dashboard tiles and usage view.

MEMORY.md checklist:
- [x] Dashboard: documents, users, search, analytics, collections
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import func
from sqlalchemy.orm import Session

from ragfabric_core.db.session import get_db
from ragfabric_core.models.document import Chunk, Collection, Document, QueryLog
from ragfabric_core.models.user import User
from ragfabric_core.store.vector_store import get_store
from ragfabric_server.deps import get_current_user

router = APIRouter()


@router.get("/overview")
def overview(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict:
    """Aggregate counts for the dashboard summary tiles."""
    return {
        "documents": db.query(Document).count(),
        "collections": db.query(Collection).count(),
        "chunks": db.query(Chunk).count(),
        "users": db.query(User).count(),
        "queries": db.query(QueryLog).count(),
        "ready_documents": db.query(Document).filter(Document.status == "ready").count(),
        # Live index size. Comparing this against ``chunks`` is the quickest way
        # to spot the vector index drifting out of sync with the database.
        "indexed_vectors": len(get_store()),
    }


@router.get("/usage")
def usage(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict:
    """Recent questions, most-indexed documents, and most-cited documents."""
    recent = db.query(QueryLog).order_by(QueryLog.created_at.desc()).limit(10).all()
    recent_queries = [
        {
            "question": log.question,
            "confidence": round(float(log.confidence), 4),
            "created_at": log.created_at.isoformat(),
        }
        for log in recent
    ]

    top_docs = (
        db.query(Document.id, Document.filename, func.count(Chunk.id).label("chunks"))
        .join(Chunk, Chunk.document_id == Document.id)
        .group_by(Document.id, Document.filename)
        .order_by(func.count(Chunk.id).desc())
        .limit(5)
        .all()
    )
    top_documents = [
        {"document_id": doc_id, "filename": filename, "chunks": chunks}
        for doc_id, filename, chunks in top_docs
    ]

    # Citation counts are tallied in Python rather than SQL: the cited ids live
    # in a JSON column and JSON aggregation is not portable between SQLite and
    # PostgreSQL, both of which this app has to run on.
    counts: dict[int, int] = {}
    for (cited,) in db.query(QueryLog.cited_document_ids).all():
        for document_id in cited or []:
            counts[document_id] = counts.get(document_id, 0) + 1

    names = dict(db.query(Document.id, Document.filename).all())
    most_cited = [
        {
            "document_id": document_id,
            "filename": names.get(document_id, "(deleted)"),
            "citations": count,
        }
        for document_id, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
    ]

    return {
        "recent_queries": recent_queries,
        "top_documents": top_documents,
        "most_cited_documents": most_cited,
    }
