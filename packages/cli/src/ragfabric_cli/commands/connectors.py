"""``ragfabric connectors``: list the configured connectors and sync them.

Connectors are configured under ``connectors:`` in ragfabric.yaml (see
docs/connectors.md). Like ``ragfabric worker`` and ``ragfabric ingest``, this
runs in-process against core and the database, not through the HTTP API.
"""

from __future__ import annotations

import threading
import time

import typer

from ragfabric_cli.commands.common import collection_by_name, session, user_by_email
from ragfabric_core.config_file import ConnectorConfig, FolderConnectorConfig
from ragfabric_core.connectors.registry import build_connector
from ragfabric_core.connectors.sync import SyncReport, sync_connector
from ragfabric_core.runtime import get_config
from ragfabric_core.telemetry.logs import bind_request_id, configure_logging

connectors_app = typer.Typer(help="Document sources: watched folders and Google Drive.")

# Seams for the tests: the clock and the wait between passes.
_now = time.monotonic


def _wait(stop: threading.Event, seconds: float) -> None:
    stop.wait(seconds)


def _where(cfg: ConnectorConfig) -> str:
    return cfg.path if isinstance(cfg, FolderConnectorConfig) else f"drive folder {cfg.folder_id}"


@connectors_app.command("list")
def list_connectors() -> None:
    """List the connectors configured in ragfabric.yaml.

    \b
    Examples:
      ragfabric connectors list
    """
    connectors = get_config().connectors
    if not connectors:
        typer.echo("no connectors configured; add a connectors: section to ragfabric.yaml")
        return
    for cfg in connectors:
        typer.echo(
            f"{cfg.name}\t{cfg.kind}\t{_where(cfg)}\tcollection {cfg.collection or '-'}\t"
            f"every {cfg.interval_seconds}s\ton_delete {cfg.on_delete}"
        )


def _run_one(cfg: ConnectorConfig) -> SyncReport:
    connector = build_connector(cfg)
    with session() as db, bind_request_id(f"connector:{cfg.name}"):
        collection_id = (
            collection_by_name(db, cfg.collection, create=True).id if cfg.collection else None
        )
        owner_id = user_by_email(db, cfg.owner).id if cfg.owner else None
        db.commit()
        return sync_connector(
            db,
            connector,
            name=cfg.name,
            collection_id=collection_id,
            owner_id=owner_id,
            on_delete=cfg.on_delete,
        )


def _print(report: SyncReport) -> None:
    typer.echo(report.summary())
    for line in report.problems:
        typer.echo(f"  {line}")
    for line in report.warnings:
        typer.echo(f"  warning: {line}")


def _changed(report: SyncReport) -> bool:
    return bool(
        report.added
        or report.updated
        or report.deleted
        or report.skipped
        or report.failed
        or report.warnings
    )


def _pass(selected: list[ConnectorConfig], *, quiet_when_idle: bool) -> bool:
    """Sync each connector once; True if every one ran without a problem."""
    clean = True
    for cfg in selected:
        try:
            report = _run_one(cfg)
        except typer.Exit:
            raise
        except Exception as exc:
            clean = False
            typer.echo(f"{cfg.name}: failed: {exc}")
            continue
        if report.failed or report.skipped:
            clean = False
        if not quiet_when_idle or _changed(report):
            _print(report)
    return clean


@connectors_app.command("sync")
def sync(
    name: str | None = typer.Argument(None, help="One connector by name; all when omitted."),
    watch: bool = typer.Option(
        False, "--watch", help="Keep running, syncing each connector every interval_seconds."
    ),
) -> None:
    """Sync configured connectors into RagFabric.

    Without --watch, one pass and exit: 0 when every file synced, 1 when a
    connector failed or a file was refused. With --watch, each connector runs
    every interval_seconds until stopped (Ctrl-C or SIGTERM).

    \b
    Examples:
      ragfabric connectors sync
      ragfabric connectors sync inbox
      ragfabric connectors sync --watch
    """
    cfg = get_config()
    configure_logging(cfg.logging.format, cfg.logging.level)
    selected = [c for c in cfg.connectors if name is None or c.name == name]
    if name is not None and not selected:
        typer.echo(f"no connector named {name}; see ragfabric connectors list")
        raise typer.Exit(code=1)
    if not selected:
        typer.echo("no connectors configured; add a connectors: section to ragfabric.yaml")
        raise typer.Exit(code=1)

    if not watch:
        if not _pass(selected, quiet_when_idle=False):
            raise typer.Exit(code=1)
        return

    from ragfabric_core.workers.runner import install_sigterm_handler

    stop = threading.Event()
    install_sigterm_handler(stop)
    due = {c.name: 0.0 for c in selected}
    typer.echo(f"watching {', '.join(due)}; Ctrl-C to stop")
    try:
        while not stop.is_set():
            now = _now()
            ready = [c for c in selected if due[c.name] <= now]
            _pass(ready, quiet_when_idle=True)
            for c in ready:
                due[c.name] = now + c.interval_seconds
            _wait(stop, max(0.0, min(due.values()) - _now()))
    except KeyboardInterrupt:
        pass
    typer.echo("stopped")
