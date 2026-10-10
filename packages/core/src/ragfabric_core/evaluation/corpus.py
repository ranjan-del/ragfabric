"""Load a corpus into the evaluation collection, once."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from sqlalchemy.orm import Session

from ragfabric_core.ingest.pipeline import ingest_document
from ragfabric_core.models.document import Collection, Document


def collection_id(db: Session, name: str, *, create: bool = False) -> int | None:
    found = db.query(Collection).filter(Collection.name == name).first()
    if found is None and create:
        found = Collection(name=name, description="RagFabric evaluation corpus")
        db.add(found)
        db.flush()
    return found.id if found is not None else None


def ingest_corpus(
    session_factory: Callable[[], Session], name: str, paths: list[Path]
) -> tuple[list[str], list[str], list[str]]:
    """Ingest ``paths`` into collection ``name``; return (ingested, skipped, failed) file names.

    A file whose name is already ``ready`` in the collection is skipped, so a
    rerun costs nothing. To re-ingest changed content, delete the documents
    first; there is no content checksum column to compare against.
    """
    ingested: list[str] = []
    skipped: list[str] = []
    failed: list[str] = []
    with session_factory() as db:
        cid = collection_id(db, name, create=True)
        db.commit()
        ready = {
            filename
            for (filename,) in db.query(Document.filename).filter(
                Document.collection_id == cid, Document.status == "ready"
            )
        }
        for path in paths:
            if path.name in ready:
                skipped.append(path.name)
                continue
            doc = ingest_document(db, filename=path.name, data=path.read_bytes(), collection_id=cid)
            (ingested if doc.status == "ready" else failed).append(path.name)
    return ingested, skipped, failed
