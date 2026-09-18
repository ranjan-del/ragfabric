from __future__ import annotations

import logging
import threading

import typer

from ragfabric_core.providers.registry import build_embedding_provider
from ragfabric_core.queue.registry import build_queue
from ragfabric_core.runtime import get_config, get_session_factory
from ragfabric_core.stores.registry import build_lexical_store, build_vector_store
from ragfabric_core.workers.runner import Worker, default_handlers


def _stores():
    cfg = get_config()
    sf = get_session_factory()
    provider = build_embedding_provider(cfg.embeddings)
    return (
        build_vector_store(cfg.vector_store, sf, embedding_model=provider.model),
        build_lexical_store(cfg.lexical_store, sf),
    )


def worker(
    once: bool = typer.Option(False, "--once", help="Process at most one job and exit."),
) -> None:
    """Run the indexing worker against the configured queue."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = get_config()
    queue = build_queue(cfg)
    if queue is None:
        typer.echo(
            "ingestion.indexing is inline; set `indexing: queue` in ragfabric.yaml to run a worker"
        )
        raise typer.Exit(code=1)
    sf = get_session_factory()
    vector_store, lexical_store = _stores()
    handlers = default_handlers(
        embedding_provider=build_embedding_provider(cfg.embeddings),
        vector_store=vector_store,
        lexical_store=lexical_store,
    )
    w = Worker(queue, sf, handlers)
    if once:
        typer.echo("processed 1 job" if w.run_once(timeout_seconds=1.0) else "queue empty")
        return
    typer.echo("worker started; waiting for jobs")
    w.run_forever(threading.Event())
