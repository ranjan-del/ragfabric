from __future__ import annotations

from pathlib import Path

import typer

from ragfabric_cli.commands.common import collection_by_name, session, user_by_email
from ragfabric_core.ingest.parser import SUPPORTED_FORMATS
from ragfabric_core.ingest.pipeline import ingest_document


def _files(path: Path, recursive: bool) -> list[Path]:
    if path.is_file():
        return [path]
    pattern = "**/*" if recursive else "*"
    return sorted(
        p
        for p in path.glob(pattern)
        if p.is_file() and p.suffix.lower().lstrip(".") in SUPPORTED_FORMATS
    )


def ingest(
    path: Path = typer.Argument(..., exists=True),
    collection: str | None = typer.Option(None, "--collection"),
    recursive: bool = typer.Option(False, "--recursive"),
    owner: str | None = typer.Option(None, "--owner", help="Email of the owning user."),
) -> None:
    """Ingest one file or every supported file in a directory."""
    files = _files(path, recursive)
    if not files:
        typer.echo("no supported files found")
        raise typer.Exit(code=1)
    failed = 0
    with session() as db:
        collection_id = collection_by_name(db, collection, create=True).id if collection else None
        owner_id = user_by_email(db, owner).id if owner else None
        db.commit()
        for file in files:
            doc = ingest_document(
                db,
                filename=file.name,
                data=file.read_bytes(),
                collection_id=collection_id,
                owner_id=owner_id,
            )
            typer.echo(
                f"{file.name}: {doc.status} ({doc.num_chunks} chunks){' ' + doc.error if doc.error else ''}"
            )
            failed += doc.status == "failed"
    typer.echo(f"{len(files) - failed} ingested, {failed} failed")
    if failed:
        raise typer.Exit(code=1)
