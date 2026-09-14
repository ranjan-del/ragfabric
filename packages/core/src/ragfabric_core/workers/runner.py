"""The worker loop: dequeue, dispatch by kind, commit or log, repeat."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable

from sqlalchemy.orm import Session

from ragfabric_core.models.document import Document
from ragfabric_core.providers.base import EmbeddingProvider
from ragfabric_core.queue.base import Job, JobQueue
from ragfabric_core.stores.base import LexicalStore, VectorStore
from ragfabric_core.workers import handlers

log = logging.getLogger(__name__)

Handler = Callable[[Session, Job], None]


def default_handlers(
    *, embedding_provider: EmbeddingProvider, vector_store: VectorStore, lexical_store: LexicalStore
) -> dict[str, Handler]:
    def _index(db: Session, job: Job) -> None:
        handlers.index_document(
            db,
            int(job.payload["document_id"]),
            embedding_provider=embedding_provider,
            vector_store=vector_store,
            lexical_store=lexical_store,
        )

    def _graph(db: Session, job: Job) -> None:
        handlers.extract_graph(db, int(job.payload["document_id"]))

    return {"index_document": _index, "extract_graph": _graph}


class Worker:
    def __init__(
        self,
        queue: JobQueue,
        session_factory: Callable[[], Session],
        handlers_by_kind: dict[str, Handler],
    ) -> None:
        self.queue = queue
        self._sf = session_factory
        self.handlers = handlers_by_kind
        self.processed = 0
        self.failed = 0

    def run_once(self, timeout_seconds: float = 1.0) -> bool:
        job = self.queue.dequeue(timeout_seconds=timeout_seconds)
        if job is None:
            return False
        handler = self.handlers.get(job.kind)
        with self._sf() as db:
            try:
                if handler is None:
                    raise KeyError(f"no handler for job kind {job.kind!r}")
                handler(db, job)
                self.processed += 1
                log.info("job %s (%s) done", job.id, job.kind)
            except Exception as exc:
                db.rollback()
                self.failed += 1
                self.processed += 1
                log.exception("job %s (%s) failed", job.id, job.kind)
                self._mark_document_failed(job, exc)
        return True

    def _mark_document_failed(self, job: Job, exc: Exception) -> None:
        document_id = job.payload.get("document_id")
        if document_id is None:
            return
        try:
            with self._sf() as fail_db:
                document = fail_db.get(Document, int(document_id))
                if document is None:
                    return
                document.status = "failed"
                document.error = f"{job.kind} failed: {exc}"[:500]
                fail_db.commit()
                log.error("document %s marked failed: %s", document_id, document.error)
        except Exception:
            log.error(
                "could not mark document %s failed after job %s error",
                document_id,
                job.id,
                exc_info=True,
            )

    def run_forever(self, stop: threading.Event, timeout_seconds: float = 1.0) -> None:
        while not stop.is_set():
            self.run_once(timeout_seconds=timeout_seconds)
