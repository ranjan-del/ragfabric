"""The worker loop: dequeue, dispatch by kind, commit or log, repeat."""

from __future__ import annotations

import logging
import signal
import threading
from collections.abc import Callable

from sqlalchemy.orm import Session

from ragfabric_core.config_file import GraphStoreConfig
from ragfabric_core.models.document import Document
from ragfabric_core.providers.base import EmbeddingProvider, LLMProvider
from ragfabric_core.queue.base import Job, JobQueue
from ragfabric_core.stores.base import LexicalStore, VectorStore
from ragfabric_core.workers import handlers

log = logging.getLogger(__name__)

Handler = Callable[[Session, Job], None]


def install_sigterm_handler(stop: threading.Event) -> None:
    """Make SIGTERM request a graceful stop instead of killing the process mid job.

    A container stop sends SIGTERM (then SIGKILL after a grace period).
    Python's default SIGTERM disposition raises ``SystemExit``, which can land
    in the middle of ``run_once`` and abandon a job partway through any of its
    several commits. Installing this handler instead only sets ``stop``:
    ``run_forever``'s loop checks it between jobs, so the job already in
    flight always finishes (success or failure, exactly as it would without a
    signal) before the loop exits and the process returns normally, i.e. exit
    code 0, not whatever exit code a raised ``SystemExit`` mid-job would leave
    behind.

    Signal handlers are only deliverable on the main thread, which is where
    the CLI's worker command runs this from.
    """

    def _handle(signum: int, frame: object) -> None:
        log.info("SIGTERM received; finishing the current job, then stopping")
        stop.set()

    signal.signal(signal.SIGTERM, _handle)


def default_handlers(
    *,
    embedding_provider: EmbeddingProvider,
    vector_store: VectorStore,
    lexical_store: LexicalStore,
    graph_settings: GraphStoreConfig,
    llm: LLMProvider | None,
) -> dict[str, Handler]:
    """The job handlers, keyed by job kind.

    ``graph_settings`` is required rather than defaulted: a default here would
    let a caller forget it and silently run every worker with the graph off
    whatever the config file says. ``llm`` is only needed when the graph is
    enabled; the embedding provider doubles as the resolver's embedder, so
    merges compare vectors from the model the deployment already configured.
    """

    def _index(db: Session, job: Job) -> None:
        handlers.index_document(
            db,
            int(job.payload["document_id"]),
            embedding_provider=embedding_provider,
            vector_store=vector_store,
            lexical_store=lexical_store,
        )

    def _graph(db: Session, job: Job) -> None:
        document_id = int(job.payload["document_id"])
        outcome = handlers.extract_graph(
            db,
            document_id,
            settings=graph_settings,
            llm=llm,
            embedder=embedding_provider,
        )
        if outcome is not None:
            log.info(
                "job %s extract_graph for document %s: %s", job.id, document_id, outcome.summary()
            )

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
