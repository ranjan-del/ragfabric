"""The ragfabric command.

Thin by design: every command calls a function in ragfabric_core or starts
ragfabric_server. No logic lives here that the API or the tests cannot reach
another way.
"""

from __future__ import annotations

import os
from pathlib import Path

import typer
from pydantic import ValidationError

from ragfabric_cli.commands import access as access_commands
from ragfabric_cli.commands import ingest as ingest_commands
from ragfabric_cli.commands import users as users_commands
from ragfabric_cli.commands.worker import worker as worker_command
from ragfabric_core import __version__
from ragfabric_core.config_file import load_config, resolve_config_path
from ragfabric_core.db import migrate
from ragfabric_core.providers.base import ProviderError
from ragfabric_core.providers.registry import build_embedding_provider, build_llm_provider

app = typer.Typer(
    help="RagFabric: self hosted, measurement first RAG platform.", no_args_is_help=True
)
db_app = typer.Typer(help="Database migrations.")
config_app = typer.Typer(help="Configuration.")
app.add_typer(db_app, name="db")
app.add_typer(config_app, name="config")
app.add_typer(users_commands.app, name="users")
app.add_typer(access_commands.groups_app, name="groups")
app.add_typer(access_commands.grants_app, name="grants")
app.add_typer(access_commands.keys_app, name="keys")
app.command("ingest")(ingest_commands.ingest)
app.command("worker")(worker_command)


def _database_url() -> str:
    from ragfabric_core.config import get_settings

    return os.environ.get("DATABASE_URL") or get_settings().database_url


@app.command()
def version() -> None:
    """Print the installed version."""
    typer.echo(f"ragfabric {__version__}")


@db_app.command()
def upgrade(revision: str = typer.Option("head", help="Target revision.")) -> None:
    """Apply migrations up to REVISION (default head)."""
    migrate.upgrade(_database_url(), revision)
    typer.echo(f"database upgraded to {revision}")


@db_app.command()
def downgrade(revision: str = typer.Option("base", help="Target revision.")) -> None:
    """Roll migrations back to REVISION (default base, which removes every table)."""
    migrate.downgrade(_database_url(), revision)
    typer.echo(f"database downgraded to {revision}")


@config_app.command("validate")
def config_validate(
    path: Path | None = typer.Option(None, "--path", help="ragfabric.yaml to validate."),
    check_providers: bool = typer.Option(
        False, "--check-providers", help="Also make one live call per provider."
    ),
) -> None:
    """Load the configuration and report the active implementation for each interface."""
    resolved = resolve_config_path(path)
    typer.echo(f"config file: {resolved if resolved else 'none (defaults)'}")
    try:
        cfg = load_config(path)
    except ValidationError as exc:
        typer.echo("configuration is invalid:")
        for err in exc.errors():
            typer.echo(f"  {'.'.join(str(p) for p in err['loc'])}: {err['msg']}")
        raise typer.Exit(code=1) from None

    problems: list[str] = []
    try:
        llm = build_llm_provider(cfg.llm)
        typer.echo(f"llm: {cfg.llm.provider} ({llm.default_model})")
    except ProviderError as exc:
        problems.append(str(exc))
    try:
        emb = build_embedding_provider(cfg.embeddings)
        typer.echo(f"embeddings: {cfg.embeddings.provider} ({emb.model})")
    except ProviderError as exc:
        problems.append(str(exc))
    typer.echo(f"reranker: {cfg.reranker.kind}")
    typer.echo(f"vector_store: {cfg.vector_store.kind}")
    typer.echo(f"lexical_store: {cfg.lexical_store.kind}")
    typer.echo(
        f"graph_store: {cfg.graph_store.kind} ({'enabled' if cfg.graph_store.enabled else 'disabled'})"
    )
    typer.echo(f"cache: {cfg.cache.kind}")
    typer.echo(f"router: {cfg.router.mode} (min_confidence {cfg.router.min_confidence})")

    if check_providers and not problems:
        from ragfabric_core.providers.base import Message

        try:
            out = llm.complete(
                [Message(role="user", content="Reply with the single word: pong")], max_tokens=8
            )
            typer.echo(
                f"llm live check: ok ({out.input_tokens} in, {out.output_tokens} out, {out.latency_ms} ms)"
            )
            vec = emb.embed(["ping"])
            typer.echo(f"embeddings live check: ok (dim {len(vec.vectors[0])})")
        except ProviderError as exc:
            problems.append(str(exc))

    if problems:
        typer.echo("problems:")
        for p in problems:
            typer.echo(f"  {p}")
        raise typer.Exit(code=1)
    typer.echo("configuration ok")


@app.command()
def init(
    force: bool = typer.Option(
        False, "--force", help="Overwrite existing .env and ragfabric.yaml."
    ),
) -> None:
    """Create .env and ragfabric.yaml from the examples, then validate."""
    import shutil
    from pathlib import Path

    for example, target in ((".env.example", ".env"), ("ragfabric.example.yaml", "ragfabric.yaml")):
        src, dst = Path(example), Path(target)
        if not src.exists():
            typer.echo(f"{example} not found in the current directory")
            raise typer.Exit(code=1)
        if dst.exists() and not force:
            typer.echo(f"{target} already exists (use --force to overwrite)")
            continue
        shutil.copyfile(src, dst)
        typer.echo(f"wrote {target}")
    from ragfabric_core.runtime import reset_config

    reset_config()
    config_validate(path=None, check_providers=False)


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Bind address."),
    port: int = typer.Option(8000, help="Port."),
    reload: bool = typer.Option(False, help="Auto reload (development only)."),
) -> None:
    """Run the HTTP API with uvicorn."""
    import uvicorn

    uvicorn.run("ragfabric_server.main:app", host=host, port=port, reload=reload)
