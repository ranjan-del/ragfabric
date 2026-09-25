"""ragfabric graph merges: inspect and undo recorded entity merges (ruling R37).

An operator command, like ``users``, ``groups``, ``grants`` and ``keys``: it
opens the configured database directly and acts on it, so running it at all
requires the database credentials only an administrator holds. There is no
HTTP endpoint for merges in this phase; the Phase 9 console owns that.

Every merge entity resolution makes is recorded as an ``EntityMerge`` row with
the evidence that justified it (Task 5). Without a way to read and reverse
those rows the record would be write-only, so ``list`` and ``show`` read them
and ``undo`` calls ``graph.resolve.unmerge`` and commits only if it succeeds.
Unmerge is globally last-in, first-out (ruling R30): an older merge cannot be
undone while a newer one is in place, and ``undo`` says which ones block it
and exits non-zero rather than attempting a partial repair.
"""

from __future__ import annotations

import json
from typing import Any

import typer

from ragfabric_cli.commands.common import session
from ragfabric_core.graph.resolve import UnmergeBlocked, UnmergeResult, unmerge
from ragfabric_core.models.graph import Entity, EntityMerge

graph_app = typer.Typer(help="Knowledge graph administration.")
merges_app = typer.Typer(help="Recorded entity merges: list, show and undo.")
graph_app.add_typer(merges_app, name="merges")

_RESULT_FIELDS = (
    "restored_entity_id",
    "survivor_entity_id",
    "moved_relationship_ids",
    "shared_relationship_ids",
    "restored_relationship_ids",
    "unrestored_relationship_ids",
    "changed_chunk_ids",
    "restored_without_sources",
)


def _survivor_name(db, entity_id: int) -> str:
    entity = db.get(Entity, entity_id)
    return entity.name if entity is not None else "?"


@merges_app.command("list")
def list_merges() -> None:
    """Every recorded merge, oldest first. Undo them newest first."""
    with session() as db:
        records = db.query(EntityMerge).order_by(EntityMerge.id).all()
        if not records:
            typer.echo("no recorded merges")
            return
        for record in records:
            typer.echo(
                f"{record.id}\t{record.method}\t{record.merged_name} ({record.merged_entity_type})"
                f" -> {record.surviving_entity_id} {_survivor_name(db, record.surviving_entity_id)}"
                f"\t{len(record.merged_source_chunk_ids)} chunk(s)\t{record.created_at:%Y-%m-%d}"
            )


def _record(db, merge_id: int) -> EntityMerge:
    record = db.get(EntityMerge, merge_id)
    if record is None:
        typer.echo(f"merge not found: {merge_id}", err=True)
        raise typer.Exit(code=1)
    return record


@merges_app.command("show")
def show_merge(merge_id: int) -> None:
    """One merge: what was merged into what, by which method, and the evidence for it."""
    with session() as db:
        record = _record(db, merge_id)
        evidence: dict[str, Any] = dict(record.evidence)
        restore: dict[str, Any] = evidence.pop("restore", {}) or {}
        typer.echo(f"merge {record.id} ({record.method}, {record.created_at:%Y-%m-%d %H:%M})")
        typer.echo(f"merged: {record.merged_name} ({record.merged_entity_type})")
        typer.echo(
            f"into: {record.surviving_entity_id} {_survivor_name(db, record.surviving_entity_id)}"
        )
        typer.echo(f"model: {record.model or 'none'}")
        typer.echo(f"merged aliases: {', '.join(record.merged_aliases) or 'none'}")
        typer.echo(f"source chunks: {sorted(record.merged_source_chunk_ids)}")
        typer.echo("evidence:")
        typer.echo(json.dumps(evidence, indent=2, sort_keys=True, default=str))
        # The restore payload is what makes the unmerge exact, not justification;
        # summarised here so an operator can see what an undo would touch.
        typer.echo(
            f"undo would restore: {len(restore.get('relationships', []))} recorded "
            f"relationship(s), {len(restore.get('dropped_self_loops', []))} dropped self "
            f"loop(s), aliases appended {restore.get('aliases_appended', [])}"
        )


@merges_app.command("undo")
def undo_merge(merge_id: int) -> None:
    """Reverse one merge. Refused while any newer merge is still in place."""
    with session() as db:
        try:
            result = unmerge(db, merge_id)
        except UnmergeBlocked as exc:
            db.rollback()
            blocking = " ".join(str(i) for i in exc.blocking_merge_ids)
            typer.echo(
                f"cannot undo merge {merge_id}: blocked by later merges: {blocking} "
                "(undo those first, newest first)",
                err=True,
            )
            raise typer.Exit(code=1) from None
        except (LookupError, ValueError) as exc:
            db.rollback()
            typer.echo(f"cannot undo merge {merge_id}: {exc}", err=True)
            raise typer.Exit(code=1) from None
        db.commit()
    typer.echo(f"undid merge {merge_id}")
    _print_result(result)


def _print_result(result: UnmergeResult) -> None:
    for field in _RESULT_FIELDS:
        typer.echo(f"{field}: {getattr(result, field)}")
