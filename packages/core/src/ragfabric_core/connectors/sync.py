"""Sync one connector into RagFabric: add what is new, update what changed, delete what is gone.

Per source, cheapest check first (design D18):

1. The ``connector_items`` row's size, modified time and source version match
   what the connector lists: unchanged, nothing is fetched.
2. Otherwise the bytes are fetched and validated with the API's own rules
   (``ingest/validate.py``). A refused file gets a row carrying the reason and
   no document, so it is not fetched again until it changes.
3. The SHA-256 of the bytes matches the row: the file was touched but not
   changed. Only the metadata is updated; the document keeps its version.
4. Otherwise the document is re-ingested in place (same id, version + 1, so
   existing citations stay valid), or ingested for the first time.

Sources the connector no longer lists are deleted with their documents when
``on_delete`` is ``delete`` (D19), and only documents this connector created:
an uploaded document, or another connector's, has no row under this name.

Two safety rules. A listing that raises aborts the run before anything is
deleted. A listing that returns nothing while documents exist deletes nothing
either, and says so: an unmounted volume or an unshared Drive folder looks
exactly like "every file was removed", and wiping a collection on that
evidence is the worse mistake.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal

from sqlalchemy.orm import Session

from ragfabric_core.config_file import LimitsConfig
from ragfabric_core.connectors.base import Connector, SourceDocument
from ragfabric_core.ingest.delete import delete_document_everywhere
from ragfabric_core.ingest.pipeline import ingest_document, reingest_document
from ragfabric_core.ingest.validate import UploadRejected, validate_upload
from ragfabric_core.models.connector import ConnectorItem
from ragfabric_core.models.document import Document
from ragfabric_core.runtime import get_config

log = logging.getLogger(__name__)

OnDelete = Literal["delete", "keep"]


@dataclass
class SyncReport:
    connector: str
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    deleted: int = 0
    skipped: int = 0
    failed: int = 0
    # One line per refused or failed source: "<source id>: <reason>".
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"{self.connector}: {self.added} added, {self.updated} updated, "
            f"{self.unchanged} unchanged, {self.deleted} deleted, {self.skipped} skipped, "
            f"{self.failed} failed"
        )


def _utc(value: datetime | None) -> datetime | None:
    """SQLite hands back naive datetimes even for timezone-aware columns; treat them as UTC."""
    if value is None:
        return None
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _metadata_matches(item: ConnectorItem, source: SourceDocument) -> bool:
    return (
        item.size_bytes == source.size_bytes
        and _utc(item.modified_at) == _utc(source.modified_at)
        and item.source_version == source.version
    )


def _remember(item: ConnectorItem, source: SourceDocument) -> None:
    item.size_bytes = source.size_bytes
    item.modified_at = source.modified_at
    item.source_version = source.version


def _sync_one(
    db: Session,
    connector: Connector,
    source: SourceDocument,
    item: ConnectorItem | None,
    *,
    name: str,
    collection_id: int | None,
    owner_id: int | None,
    limits: LimitsConfig,
    report: SyncReport,
) -> None:
    if item is not None and _metadata_matches(item, source):
        report.unchanged += 1
        return

    try:
        data = connector.fetch(source.source_id)
    except Exception as exc:
        report.failed += 1
        report.problems.append(f"{source.source_id}: could not fetch ({exc})")
        log.warning("connector %s could not fetch %s: %s", name, source.source_id, exc)
        return

    if item is None:
        item = ConnectorItem(connector=name, source_id=source.source_id)
        db.add(item)

    try:
        validate_upload(source.name, data, limits)
    except UploadRejected as exc:
        _remember(item, source)
        item.rejected_reason = exc.reason
        db.commit()
        report.skipped += 1
        report.problems.append(f"{source.source_id}: {exc.reason}")
        log.warning("connector %s refused %s: %s", name, source.source_id, exc.reason)
        return

    content_hash = hashlib.sha256(data).hexdigest()
    document = db.get(Document, item.document_id) if item.document_id is not None else None
    if document is not None and item.content_hash == content_hash:
        _remember(item, source)
        item.rejected_reason = None
        db.commit()
        report.unchanged += 1
        return

    if document is not None:
        document = reingest_document(
            db, document, filename=source.name, data=data, content_type=source.content_type
        )
        report.updated += 1
    else:
        document = ingest_document(
            db,
            filename=source.name,
            data=data,
            content_type=source.content_type,
            collection_id=collection_id,
            owner_id=owner_id,
        )
        report.added += 1
    if document.status == "failed":
        report.problems.append(f"{source.source_id}: ingestion failed ({document.error})")

    item.document_id = document.id
    item.content_hash = content_hash
    item.rejected_reason = None
    _remember(item, source)
    db.commit()


def sync_connector(
    db: Session,
    connector: Connector,
    *,
    name: str,
    collection_id: int | None = None,
    owner_id: int | None = None,
    on_delete: OnDelete = "delete",
    limits: LimitsConfig | None = None,
) -> SyncReport:
    """One pass over ``connector``. ``name`` keys its state in ``connector_items``."""
    limits = limits or get_config().limits
    report = SyncReport(connector=name)
    items = {
        item.source_id: item
        for item in db.query(ConnectorItem).filter(ConnectorItem.connector == name).all()
    }
    # Listed in full before anything changes: a listing that fails part way
    # raises here, and nothing below runs, so nothing is deleted on a partial view.
    sources = list(connector.list_documents())
    listed = {source.source_id for source in sources}

    for source in sources:
        _sync_one(
            db,
            connector,
            source,
            items.get(source.source_id),
            name=name,
            collection_id=collection_id,
            owner_id=owner_id,
            limits=limits,
            report=report,
        )

    gone = [item for source_id, item in items.items() if source_id not in listed]
    if gone and not sources:
        report.warnings.append(
            f"{name} listed nothing while {len(gone)} source(s) are recorded; nothing was "
            "deleted. If the source really is empty now, delete the documents by hand."
        )
        log.warning("%s", report.warnings[-1])
        return report
    if on_delete == "keep":
        return report
    for item in gone:
        item_id, document_id = item.id, item.document_id
        if document_id is not None:
            delete_document_everywhere(db, document_id)
            report.deleted += 1
        # The cascade has already removed a row whose document was deleted;
        # a refused source's row has no document and is removed here.
        db.expire_all()
        remaining = db.get(ConnectorItem, item_id)
        if remaining is not None:
            db.delete(remaining)
            db.commit()
    return report
