"""ragfabric reconcile: retry documents stuck mid fan-out.

index_document writes the vector store and the lexical store as two separate
commits (workers/handlers.py), so a hard crash between them leaves a document
stuck in "indexing" forever with no exception for anything to have caught.
This command is the operator-triggered retry: it finds every document stuck
in "indexing" past the grace period and re-runs indexing for it, which is
safe because index_document is idempotent (see workers/handlers.py).
"""

from __future__ import annotations

import typer

from ragfabric_cli.commands.common import session
from ragfabric_core.providers.registry import build_embedding_provider
from ragfabric_core.runtime import get_config, get_session_factory
from ragfabric_core.stores.registry import build_lexical_store, build_vector_store
from ragfabric_core.workers.handlers import reconcile_stuck_indexing


def reconcile(
    older_than_seconds: int = typer.Option(
        300,
        "--older-than",
        min=0,
        help="Only retry documents that have been stuck in 'indexing' for at least this long.",
    ),
) -> None:
    """Retry any document left stuck in 'indexing' by a crashed fan out."""
    cfg = get_config()
    sf = get_session_factory()
    provider = build_embedding_provider(cfg.embeddings)
    vector_store = build_vector_store(cfg.vector_store, sf, embedding_model=provider.model)
    lexical_store = build_lexical_store(cfg.lexical_store, sf)
    with session() as db:
        count = reconcile_stuck_indexing(
            db,
            embedding_provider=provider,
            vector_store=vector_store,
            lexical_store=lexical_store,
            older_than_seconds=older_than_seconds,
        )
    typer.echo(f"reconciled {count} document(s) stuck in indexing")
