from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import typer
from rich.markup import escape
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn

from ragfabric_cli.commands.common import collection_by_name, session, user_by_email
from ragfabric_cli.ui import console
from ragfabric_core.ingest.parser import SUPPORTED_FORMATS
from ragfabric_core.ingest.pipeline import ingest_document
from ragfabric_core.models.document import Document


def _files(path: Path, recursive: bool) -> list[Path]:
    if path.is_file():
        return [path]
    pattern = "**/*" if recursive else "*"
    return sorted(
        p
        for p in path.glob(pattern)
        if p.is_file() and p.suffix.lower().lstrip(".") in SUPPORTED_FORMATS
    )


def _describe(file: Path, doc: Document) -> str:
    return (
        f"{file.name}: {doc.status} ({doc.num_chunks} chunks){' ' + doc.error if doc.error else ''}"
    )


def ingest_files(
    files: list[Path],
    *,
    collection: str | None,
    owner: str | None,
    on_file: Callable[[Path, Document], None] | None = None,
    quiet: bool = False,
) -> tuple[int, int]:
    """Ingest ``files`` and return ``(ingested, failed)``.

    ``collection`` is a collection name (created if missing) and ``owner`` a user email, both
    resolved inside the session this function opens. On a terminal a progress bar is shown and
    only failed files are listed, under it; piped, every file gets its plain line.
    With ``quiet=True`` nothing is printed and no bar is made; ``on_file`` still fires.
    """
    failed = 0
    with session() as db:
        collection_id = collection_by_name(db, collection, create=True).id if collection else None
        owner_id = user_by_email(db, owner).id if owner else None
        db.commit()
        rich = not quiet and console.is_rich()
        progress = (
            Progress(
                TextColumn("{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                console=console.get_console(),
                transient=True,
            )
            if rich
            else None
        )
        task = progress.add_task("", total=len(files)) if progress else None
        if progress:
            progress.start()
        try:
            for file in files:
                if progress and task is not None:
                    progress.update(task, description=escape(file.name))
                doc = ingest_document(
                    db,
                    filename=file.name,
                    data=file.read_bytes(),
                    collection_id=collection_id,
                    owner_id=owner_id,
                )
                if doc.status == "failed":
                    failed += 1
                if progress and task is not None:
                    if doc.status == "failed":
                        progress.console.print(escape(_describe(file, doc)), soft_wrap=True)
                    progress.advance(task)
                elif not quiet:
                    typer.echo(_describe(file, doc))
                if on_file is not None:
                    on_file(file, doc)
        finally:
            if progress:
                progress.stop()
    return len(files) - failed, failed


def ingest(
    path: Path = typer.Argument(..., exists=True),
    collection: str | None = typer.Option(None, "--collection"),
    recursive: bool = typer.Option(False, "--recursive"),
    owner: str | None = typer.Option(None, "--owner", help="Email of the owning user."),
) -> None:
    """Ingest one file or every supported file in a directory.

    \b
    Examples:
      ragfabric ingest ./docs --recursive
      ragfabric ingest handbook.pdf --collection hr
    """
    files = _files(path, recursive)
    if not files:
        typer.echo("no supported files found")
        raise typer.Exit(code=1)
    ingested, failed = ingest_files(files, collection=collection, owner=owner)
    typer.echo(f"{ingested} ingested, {failed} failed")
    if failed:
        raise typer.Exit(code=1)
