"""ragfabric doctor: check the environment and say what to fix."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

import typer

from ragfabric_cli.commands.ask import DEFAULT_URL
from ragfabric_cli.ui.panels import CheckView, render_checks
from ragfabric_core import diagnostics


def doctor(
    url: str | None = typer.Option(
        None, "--url", help=f"Server URL (env RAGFABRIC_URL, default {DEFAULT_URL})."
    ),
    no_network: bool = typer.Option(
        False, "--no-network", help="Skip the checks that call a model provider or the server."
    ),
    as_json: bool = typer.Option(False, "--json", help="Print the results as JSON."),
) -> None:
    """Check Python, config, database, migrations, providers and the server.

    Exits 1 when any check fails. A check that did not run is reported as skip.

    \b
    Examples:
      ragfabric doctor
      ragfabric doctor --no-network
      ragfabric doctor --json
    """
    from ragfabric_core.config import get_settings

    db_url = os.environ.get("DATABASE_URL") or get_settings().database_url
    config_env = os.environ.get("RAGFABRIC_CONFIG")
    results = diagnostics.run_all(
        config_path=Path(config_env) if config_env else None,
        db_url=db_url,
        server_url=url or os.environ.get("RAGFABRIC_URL") or DEFAULT_URL,
        network=not no_network,
    )
    if as_json:
        typer.echo(json.dumps([asdict(r) for r in results]))
    else:
        render_checks([CheckView(r.name, r.status, r.detail, r.fix) for r in results])
    if any(r.status == "fail" for r in results):
        raise typer.Exit(code=1)
