"""ragfabric reindex: re-embed the corpus under the active embedding model."""

from __future__ import annotations

import typer

from ragfabric_cli.commands.common import session
from ragfabric_core.ingest.reindex import reindex_all
from ragfabric_core.providers.registry import build_embedding_provider
from ragfabric_core.runtime import get_config, get_session_factory
from ragfabric_core.stores.registry import build_lexical_store, build_vector_store


def reindex(
    batch_size: int = typer.Option(64, "--batch-size", min=1, help="Chunks per embedding call."),
    document_id: int | None = typer.Option(None, "--document-id", help="Limit to one document."),
    yes: bool = typer.Option(False, "--yes", help="Do not ask for confirmation."),
) -> None:
    """Re-embed every chunk with the configured embedding model."""
    cfg = get_config()
    provider = build_embedding_provider(cfg.embeddings)
    typer.echo(f"This replaces every stored vector using {provider.model} ({provider.dim} dims).")
    if not yes and not typer.confirm("Continue?"):
        typer.echo("aborted")
        raise typer.Exit(1)

    sf = get_session_factory()
    vector_store = build_vector_store(cfg.vector_store, sf, embedding_model=provider.model)
    lexical_store = build_lexical_store(cfg.lexical_store, sf)

    def progress(done: int, total: int) -> None:
        typer.echo(f"  {done}/{total}")

    with session() as db:
        count = reindex_all(
            db,
            embedding_provider=provider,
            vector_store=vector_store,
            lexical_store=lexical_store,
            batch_size=batch_size,
            document_id=document_id,
            on_progress=progress,
        )
    typer.echo(f"re-embedded {count} chunks with {provider.model}")
