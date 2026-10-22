"""Read one stored chunk with the text around it.

A citation carries a short snippet. For a PDF, Word or PowerPoint file the original
cannot be read back as text in the browser, so the source viewer asks this route for
the cited chunk and its neighbours from the chunks stored at ingestion.

The access predicate is the one every other read of a document uses: the chunk's
document must be allowed for the caller, and a chunk that does not exist and a chunk
the caller may not read are the same 404, so the route never confirms that a
restricted chunk exists. The neighbours belong to the same document, so one check
covers them all.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from ragfabric_core.auth.principal import AccessFilter, Principal
from ragfabric_core.db.session import get_db
from ragfabric_core.models.document import Chunk, Document
from ragfabric_server.deps import get_access_filter, get_principal
from ragfabric_server.schemas.document import ChunkContext, ChunkPiece

router = APIRouter()

_NOT_FOUND = "Chunk not found."


def _piece(chunk: Chunk) -> ChunkPiece:
    return ChunkPiece(
        chunk_id=chunk.id,
        chunk_index=chunk.chunk_index,
        page=chunk.page,
        text=chunk.text,
    )


def _without_overlap(neighbour: Chunk, cited: Chunk, *, before: bool) -> ChunkPiece:
    """Drop the characters a neighbour shares with the cited chunk, when offsets allow.

    Chunks overlap by ``ingestion.chunk_overlap`` characters. The offsets are used only
    when they agree with the stored text length; otherwise the neighbour is returned whole
    rather than cut by an offset that cannot be trusted.
    """
    piece = _piece(neighbour)
    if neighbour.char_end - neighbour.char_start != len(neighbour.text):
        return piece
    if before and neighbour.char_end > cited.char_start >= neighbour.char_start:
        piece.text = neighbour.text[: cited.char_start - neighbour.char_start]
    elif not before and neighbour.char_start < cited.char_end <= neighbour.char_end:
        piece.text = neighbour.text[cited.char_end - neighbour.char_start :]
    return piece


@router.get("/{chunk_id}/context", response_model=ChunkContext)
def get_chunk_context(
    chunk_id: int,
    neighbours: int = Query(default=1, ge=0, le=5, description="Chunks to return on each side."),
    db: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
    access: AccessFilter = Depends(get_access_filter),
) -> ChunkContext:
    """The cited chunk with up to ``neighbours`` chunks of the same document on each side."""
    chunk = db.get(Chunk, chunk_id)
    document = db.get(Document, chunk.document_id) if chunk is not None else None
    if chunk is None or document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=_NOT_FOUND)
    if not access.allows(document.id, document.collection_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=_NOT_FOUND)

    around = (
        db.query(Chunk)
        .filter(
            Chunk.document_id == chunk.document_id,
            Chunk.chunk_index >= chunk.chunk_index - neighbours,
            Chunk.chunk_index <= chunk.chunk_index + neighbours,
            Chunk.id != chunk.id,
        )
        .order_by(Chunk.chunk_index)
        .all()
    )
    return ChunkContext(
        chunk_id=chunk.id,
        document_id=document.id,
        filename=document.filename,
        section=chunk.section,
        chunk=_piece(chunk),
        before=[
            _without_overlap(c, chunk, before=True)
            for c in around
            if c.chunk_index < chunk.chunk_index
        ],
        after=[
            _without_overlap(c, chunk, before=False)
            for c in around
            if c.chunk_index > chunk.chunk_index
        ],
    )
