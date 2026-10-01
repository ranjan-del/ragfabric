"""ragfabric quickstart: from nothing to a first cited answer.

Six steps, each of which checks whether it is already done and skips if so:
write the config files, choose a model, choose a database, migrate, ingest the
packaged samples, and ask the sample question through a temporary server.

Two rules shape every step. Nothing here prints a secret: .env is written and
read but never echoed, and any URL shown goes through mask_urls_in. And nothing
here leaves a process behind: the temporary server is stopped on success, on
any exception and on Ctrl-C, killed if it ignores the request to stop.
"""

from __future__ import annotations

import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import httpx
import typer

from ragfabric_cli.templates import SAMPLE_QUESTION, sample_paths, template_path
from ragfabric_cli.ui import console
from ragfabric_cli.ui.console import mask_urls_in
from ragfabric_cli.ui.panels import render_answer, render_next_steps

OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"
OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_CHAT_MODEL = "llama3.2:3b"
OLLAMA_EMBED_MODEL = "nomic-embed-text"
# Migration 0004 pins the pgvector column at 768, the nomic-embed-text width.
PINNED_DIM = 768
HEALTH_TIMEOUT_S = 30.0
STOP_TIMEOUT_S = 5.0
DB_WAIT_S = 60.0

UPGRADE_HINT = (
    f"ollama pull {OLLAMA_CHAT_MODEL} && ollama pull {OLLAMA_EMBED_MODEL} && "
    "ragfabric quickstart --force && ragfabric reindex --yes"
)


class QuickstartError(Exception):
    """A step failed in a way the user can act on. The message says how."""


@dataclass(frozen=True)
class ModelChoice:
    kind: Literal["ollama", "openai", "anthropic", "offline"]
    llm: dict
    embeddings: dict
    reason: str


@dataclass
class Context:
    dir: Path
    force: bool
    yes: bool
    docker: bool
    model_check: bool
    config_written: bool = False
    env_written: bool = False
    db_url: str = ""
    choice: ModelChoice | None = None
    child_env: dict[str, str] = field(default_factory=dict)


def _say(message: str) -> None:
    typer.echo(mask_urls_in(message))


def _done(step: str, detail: str) -> None:
    _say(f"{step}: {detail} (already done)")


# Model ----------------------------------------------------------------------


def _ollama_models(timeout: float = 2.0) -> list[str] | None:
    """The model names a local Ollama serves, or None when it does not answer."""
    try:
        res = httpx.get(OLLAMA_TAGS_URL, timeout=timeout)
        res.raise_for_status()
        return [m.get("name", "") for m in res.json().get("models", [])]
    except (httpx.HTTPError, ValueError):
        return None


def _configured_dim(cfg_path: Path | None) -> int:
    if cfg_path is None or not Path(cfg_path).is_file():
        return PINNED_DIM
    from ragfabric_core.config_file import load_config

    return load_config(cfg_path).embeddings.dim or PINNED_DIM


def _offline_embeddings(dim: int) -> dict:
    return {"provider": "offline", "model": None, "dim": dim, "base_url": None}


def choose_model(
    cfg_path: Path | None, env: Mapping[str, str], *, probe: bool = True
) -> ModelChoice:
    """Pick the best model that is available right now, without downloading anything.

    Ollama with a chat model and nomic-embed-text, then OPENAI_API_KEY, then
    ANTHROPIC_API_KEY (which has no embedding model, so embeddings go offline at
    the configured dim), then offline for both. Only whether a key is set is
    read, never its value.
    """
    notes: list[str] = []
    if probe:
        models = _ollama_models(timeout=2.0)
        if models is None:
            notes.append("Ollama is not running on localhost:11434")
        else:
            chat = [m for m in models if "embed" not in m]
            embed = [m for m in models if m.split(":")[0] == OLLAMA_EMBED_MODEL]
            if chat and embed:
                model = OLLAMA_CHAT_MODEL if OLLAMA_CHAT_MODEL in chat else chat[0]
                return ModelChoice(
                    "ollama",
                    {"provider": "ollama", "model": model, "base_url": OLLAMA_BASE_URL},
                    {
                        "provider": "ollama",
                        "model": OLLAMA_EMBED_MODEL,
                        "dim": PINNED_DIM,
                        "base_url": OLLAMA_BASE_URL,
                    },
                    f"Ollama is running with {model} and {OLLAMA_EMBED_MODEL}",
                )
            missing = []
            if not chat:
                missing.append(f"ollama pull {OLLAMA_CHAT_MODEL}")
            if not embed:
                missing.append(f"ollama pull {OLLAMA_EMBED_MODEL}")
            notes.append(
                "Ollama is running but is missing a model (run: " + "; ".join(missing) + ")"
            )
    else:
        notes.append("Ollama was not checked (--no-model-check)")

    if env.get("OPENAI_API_KEY"):
        return ModelChoice(
            "openai",
            {"provider": "openai", "model": "gpt-5.4-mini", "base_url": None},
            {
                "provider": "openai",
                "model": "text-embedding-3-small",
                "dim": 1536,
                "base_url": None,
            },
            "; ".join([*notes, "OPENAI_API_KEY is set"]),
        )
    if env.get("ANTHROPIC_API_KEY"):
        dim = _configured_dim(cfg_path)
        return ModelChoice(
            "anthropic",
            {"provider": "anthropic", "model": "claude-sonnet-5", "base_url": None},
            _offline_embeddings(dim),
            "; ".join(
                [
                    *notes,
                    "ANTHROPIC_API_KEY is set (Anthropic has no embedding model, "
                    f"so embeddings are offline hashing at dim {dim})",
                ]
            ),
        )
    dim = _configured_dim(cfg_path)
    return ModelChoice(
        "offline",
        {"provider": "offline", "model": None, "base_url": None},
        _offline_embeddings(dim),
        "; ".join(
            [
                *notes,
                "no OPENAI_API_KEY or ANTHROPIC_API_KEY is set, so offline mode: answers are "
                "extractive (sentences lifted from your documents, no model writes them) "
                "and retrieval uses hashing embeddings, not semantic ones",
            ]
        ),
    )


def _yaml_value(value: object) -> str:
    return "null" if value is None else str(value)


def patch_config(text: str, choice: ModelChoice, *, cache_kind: str | None) -> str:
    """Replace the known key lines of the packaged template, keeping every comment.

    Only ``llm.provider/model/base_url``, ``embeddings.provider/model/dim/base_url``
    and ``cache.kind`` are touched, and only their value: the indentation and
    the trailing comment stay where they were. A YAML round trip would lose the
    comments, which are most of the template's documentation.
    """
    wanted: dict[str, dict[str, object]] = {
        "llm": {k: choice.llm[k] for k in ("provider", "model", "base_url")},
        "embeddings": {k: choice.embeddings[k] for k in ("provider", "model", "dim", "base_url")},
    }
    if cache_kind is not None:
        wanted["cache"] = {"kind": cache_kind}
    line_re = re.compile(r"^(  )(\w+):( *)(\S+)(\s*#.*)?$")
    section = None
    out = []
    for line in text.splitlines(keepends=True):
        body = line.rstrip("\n")
        top = re.match(r"^(\w+):", body)
        if top:
            section = top.group(1)
        match = line_re.match(body)
        if match and section in wanted and match.group(2) in wanted[section]:
            indent, key, _, old, comment = match.groups()
            value = _yaml_value(wanted[section][key])
            head = f"{indent}{key}: {value}"
            if comment:
                width = len(f"{indent}{key}: {old}")
                head = head.ljust(width) + comment
            line = head + ("\n" if line.endswith("\n") else "")
        out.append(line)
    return "".join(out)


# Steps ----------------------------------------------------------------------


def _step_config(ctx: Context) -> None:
    for template, target in (("env.example", ".env"), ("ragfabric.example.yaml", "ragfabric.yaml")):
        dst = ctx.dir / target
        if dst.exists() and not ctx.force:
            _done("config", f"{target} exists, keeping it (use --force to overwrite)")
            continue
        shutil.copyfile(template_path(template), dst)
        if target == "ragfabric.yaml":
            ctx.config_written = True
        else:
            ctx.env_written = True
        _say(f"config: wrote {target}")


def _step_model(ctx: Context) -> None:
    cfg_path = ctx.dir / "ragfabric.yaml"
    if not ctx.config_written:
        _done("model", "ragfabric.yaml was kept, so its llm and embeddings are used as written")
        return
    choice = choose_model(cfg_path, os.environ, probe=ctx.model_check)
    ctx.choice = choice
    sqlite = ctx.db_url.startswith("sqlite")
    if not sqlite and choice.embeddings["dim"] != PINNED_DIM:
        # PostgreSQL pins the vector column at 768 (migration 0004).
        choice = ModelChoice(
            choice.kind,
            choice.llm,
            _offline_embeddings(PINNED_DIM),
            choice.reason + f"; PostgreSQL pins vectors at {PINNED_DIM}, so embeddings "
            "are offline hashing at that dim",
        )
        ctx.choice = choice
    text = patch_config(cfg_path.read_text(), choice, cache_kind="memory" if sqlite else None)
    cfg_path.write_text(text)
    llm = choice.llm["model"] or "extractive"
    emb = choice.embeddings["model"] or f"hashing, dim {choice.embeddings['dim']}"
    _say(f"model: {choice.kind} (llm {llm}; embeddings {emb})")
    _say(f"  why: {choice.reason}")
    if choice.kind == "offline":
        _say("  offline mode: answers are extractive and limited.")
        _say(f"  upgrade later with: {UPGRADE_HINT}")


def _read_env_value(path: Path, key: str) -> str | None:
    if not path.is_file():
        return None
    for line in path.read_text().splitlines():
        name, sep, value = line.partition("=")
        if sep and name.strip() == key:
            return value.strip().strip("'\"") or None
    return None


def _set_env_value(path: Path, key: str, value: str) -> None:
    lines = path.read_text().splitlines(keepends=True) if path.is_file() else []
    for i, line in enumerate(lines):
        if line.partition("=")[0].strip() == key:
            lines[i] = f"{key}={value}\n"
            break
    else:
        lines.append(f"{key}={value}\n")
    path.write_text("".join(lines))


def _docker_available() -> bool:
    try:
        return (
            subprocess.run(
                ["docker", "info"], capture_output=True, timeout=15, check=False
            ).returncode
            == 0
        )
    except (OSError, subprocess.TimeoutExpired):
        return False


def _compose_up(dir: Path) -> None:
    done = subprocess.run(
        ["docker", "compose", "up", "-d", "postgres", "redis"],
        cwd=dir,
        capture_output=True,
        text=True,
        check=False,
    )
    if done.returncode != 0:
        raise QuickstartError(
            "docker compose up -d postgres redis failed: "
            + mask_urls_in(done.stderr.strip()[-500:])
        )


def _wait_for_database(db_url: str, timeout: float = DB_WAIT_S) -> None:
    from ragfabric_core.diagnostics import check_database

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if check_database(db_url).status == "pass":
            return
        time.sleep(1)
    raise QuickstartError(
        f"the database at {mask_urls_in(db_url)} did not accept connections in {timeout:.0f}s"
    )


def _docker_url(dir: Path) -> str:
    """The URL of the compose postgres, from the same POSTGRES_* values compose reads from .env."""
    from urllib.parse import quote

    env = dir / ".env"
    user = _read_env_value(env, "POSTGRES_USER") or "ragfabric"
    password = _read_env_value(env, "POSTGRES_PASSWORD") or "ragfabric"
    name = _read_env_value(env, "POSTGRES_DB") or "ragfabric"
    return f"postgresql+psycopg://{quote(user)}:{quote(password)}@localhost:5432/{quote(name)}"


def _sqlite_url(dir: Path) -> str:
    return f"sqlite:///{dir / 'ragfabric.db'}"


def _absolute_sqlite(url: str, dir: Path) -> str:
    """Resolve a relative SQLite path against ``dir``, where the server will run."""
    prefix = "sqlite:///"
    if not url.startswith(prefix):
        return url
    path = url[len(prefix) :]
    if not path or path == ":memory:" or Path(path).is_absolute():
        return url
    return prefix + str((dir / path).resolve())


def _use_database(ctx: Context, url: str, what: str) -> None:
    """Record the database, writing it to .env only when this run wrote .env."""
    ctx.db_url = url
    env_path = ctx.dir / ".env"
    if ctx.env_written:
        _set_env_value(env_path, "DATABASE_URL", url)
        _say(f"database: {what} at {url}")
    elif _absolute_sqlite(_read_env_value(env_path, "DATABASE_URL") or "", ctx.dir) == url:
        _done("database", f"{what} at {url}")
    else:
        _say(
            f"database: {what} at {url}. .env was kept, so its DATABASE_URL is unchanged; "
            "rerun with --force to write it"
        )


def _step_database(ctx: Context) -> None:
    if ctx.docker:
        if _docker_available():
            compose = ctx.dir / "docker-compose.yml"
            if not compose.exists() or ctx.force:
                shutil.copyfile(template_path("docker-compose.yml"), compose)
                _say("database: wrote docker-compose.yml")
            with _spinner("database: starting postgres and redis with docker compose"):
                _compose_up(ctx.dir)
                url = _docker_url(ctx.dir)
                _wait_for_database(url)
            _use_database(ctx, url, "PostgreSQL")
            return
        _say(
            "database: Docker is not reachable (docker info failed). Start Docker Desktop, "
            "OrbStack or the docker service, then rerun with --docker."
        )
        if not ctx.yes and not typer.confirm("Use SQLite instead?", default=True):
            raise typer.Exit(1)
        _say("database: falling back to SQLite")
        _use_database(ctx, _sqlite_url(ctx.dir), "SQLite")
        return

    existing = _read_env_value(ctx.dir / ".env", "DATABASE_URL")
    if existing and not ctx.env_written:
        ctx.db_url = _absolute_sqlite(existing, ctx.dir)
        _done("database", f"using DATABASE_URL from .env ({ctx.db_url})")
        return
    _use_database(ctx, _sqlite_url(ctx.dir), "SQLite")


@contextmanager
def _spinner(message: str) -> Iterator[None]:
    if console.is_rich():
        with console.get_console().status(mask_urls_in(message)):
            yield
    else:
        _say(message)
        yield


@contextmanager
def _bound_runtime(ctx: Context) -> Iterator[None]:
    """Point the in-process runtime at this directory for the duration of a step.

    The engine in ragfabric_core.db.session is built at import, before this
    command knew which directory or database it would use, and ingestion
    reads ragfabric.yaml and the uploads directory through the runtime. This
    swaps all three for the step and restores them afterwards.
    """
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from ragfabric_core.db import session as session_module
    from ragfabric_core.runtime import reset_config

    saved_env = {k: os.environ.get(k) for k in ("DATABASE_URL", "RAGFABRIC_CONFIG")}
    saved_factory = session_module.SessionLocal
    saved_cwd = Path.cwd()
    kwargs = (
        {"connect_args": {"check_same_thread": False}} if ctx.db_url.startswith("sqlite") else {}
    )
    engine = create_engine(ctx.db_url, future=True, **kwargs)
    try:
        os.environ["DATABASE_URL"] = ctx.db_url
        os.environ["RAGFABRIC_CONFIG"] = str(ctx.dir / "ragfabric.yaml")
        session_module.SessionLocal = sessionmaker(
            bind=engine, autoflush=False, autocommit=False, future=True
        )
        os.chdir(ctx.dir)
        reset_config()
        yield
    finally:
        os.chdir(saved_cwd)
        session_module.SessionLocal = saved_factory
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        reset_config()
        engine.dispose()


def _step_migrate(ctx: Context) -> None:
    from ragfabric_core.db import migrate

    head = migrate.head_revision()
    try:
        current = migrate.current_revision(ctx.db_url)
    except Exception as exc:
        raise QuickstartError(f"cannot reach the database at {mask_urls_in(ctx.db_url)}") from exc
    if current == head:
        _done("migrate", f"database at {head}")
        return
    with _spinner("migrate: applying migrations"):
        migrate.upgrade(ctx.db_url)
    _say(f"migrate: database upgraded to {head}")


# The tracks merge replaces this helper with commands.ingest.ingest_files.
def ingest_files(
    files: list[Path],
    *,
    collection: str | None,
    owner: str | None,
    on_file: Callable[[Path, object], None] | None = None,
) -> tuple[int, int]:
    """Ingest each file the way ``ragfabric ingest`` does. Returns (ingested, failed)."""
    from ragfabric_cli.commands.common import collection_by_name, session, user_by_email
    from ragfabric_core.ingest.pipeline import ingest_document

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
            if on_file is not None:
                on_file(file, doc)
            failed += doc.status == "failed"
    return len(files) - failed, failed


def _present_filenames(names: list[str]) -> set[str]:
    from ragfabric_cli.commands.common import session
    from ragfabric_core.models.document import Document

    with session() as db:
        rows = db.query(Document.filename).filter(Document.filename.in_(names)).all()
    return {row[0] for row in rows}


def _step_ingest(ctx: Context) -> None:
    samples = sample_paths()
    with _bound_runtime(ctx):
        present = _present_filenames([p.name for p in samples])
        missing = [p for p in samples if p.name not in present]
        if not missing:
            _done("ingest", f"{len(samples)} sample documents present")
            return

        def report(file: Path, doc) -> None:
            error = f" {doc.error}" if doc.error else ""
            _say(f"ingest: {file.name}: {doc.status} ({doc.num_chunks} chunks){error}")

        ingested, failed = ingest_files(missing, collection=None, owner=None, on_file=report)
    _say(f"ingest: {ingested} ingested, {failed} failed")
    if failed:
        raise QuickstartError("some sample documents failed to ingest; see the lines above")


# Ask ------------------------------------------------------------------------


def free_port() -> int:
    """A TCP port on 127.0.0.1 that nothing is listening on right now."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _server_command(port: int) -> list[str]:
    return [
        sys.executable,
        "-m",
        "uvicorn",
        "ragfabric_server.main:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]


def _stop(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=STOP_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def _log_tail(log, limit: int = 1500) -> str:
    log.flush()
    log.seek(0)
    return mask_urls_in(log.read().decode("utf-8", "replace")[-limit:])


@contextmanager
def temporary_server(dir: Path, port: int, env: Mapping[str, str] | None = None) -> Iterator[str]:
    """Run the API in a child process until the block ends, then always stop it.

    Yields the base URL once /health answers 200. The process is stopped in
    ``finally``, so success, any exception and KeyboardInterrupt all reach it;
    one that ignores terminate for STOP_TIMEOUT_S seconds is killed.
    """
    url = f"http://127.0.0.1:{port}"
    with tempfile.TemporaryFile() as log:
        proc = subprocess.Popen(
            _server_command(port),
            cwd=dir,
            env=dict(env) if env is not None else None,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + HEALTH_TIMEOUT_S
            while True:
                if proc.poll() is not None:
                    raise QuickstartError(
                        f"the temporary server exited with code {proc.returncode}:\n"
                        + _log_tail(log)
                    )
                try:
                    if httpx.get(f"{url}/health", timeout=1.0).status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                if time.monotonic() > deadline:
                    raise QuickstartError(
                        f"the temporary server did not become healthy in {HEALTH_TIMEOUT_S:.0f}s:\n"
                        + _log_tail(log)
                    )
                time.sleep(0.2)
            yield url
        finally:
            _stop(proc)


def _child_env(ctx: Context) -> dict[str, str]:
    env = dict(os.environ)
    env["DATABASE_URL"] = ctx.db_url
    env["RAGFABRIC_CONFIG"] = str(ctx.dir / "ragfabric.yaml")
    return env


def _admin_credentials(ctx: Context) -> tuple[str, str]:
    from ragfabric_core.config import Settings

    settings = Settings(_env_file=str(ctx.dir / ".env"))
    if not (settings.first_admin_email and settings.first_admin_password):
        raise QuickstartError(
            "no bootstrap admin to ask as: set FIRST_ADMIN_EMAIL and FIRST_ADMIN_PASSWORD in .env"
        )
    return settings.first_admin_email, settings.first_admin_password


def _step_ask(ctx: Context) -> None:
    from ragfabric_sdk import Client

    email, password = _admin_credentials(ctx)
    port = free_port()
    _say(f"ask: starting a temporary server on 127.0.0.1:{port}")
    with temporary_server(ctx.dir, port, env=_child_env(ctx)) as url:
        res = httpx.post(
            f"{url}/api/auth/login", data={"username": email, "password": password}, timeout=10
        )
        if res.status_code != 200:
            raise QuickstartError(
                "could not sign in as the bootstrap admin from .env "
                f"(HTTP {res.status_code}); check FIRST_ADMIN_EMAIL and FIRST_ADMIN_PASSWORD"
            )
        token = res.json()["access_token"]
        _say(f"ask: {SAMPLE_QUESTION}")
        with Client(url, token=token, timeout=120) as client:
            answer = client.ask(SAMPLE_QUESTION)
    render_answer(
        answer.answer,
        [citation.model_dump() for citation in answer.citations],
        strategy=answer.strategy,
        router=answer.router.model_dump() if answer.router is not None else None,
        fallback_from=answer.fallback_from,
        subgraph=answer.subgraph.model_dump() if answer.subgraph is not None else None,
        dropped_claims=[],
        dropped_relationship_claims=[
            claim.model_dump() for claim in answer.dropped_relationship_claims
        ],
    )
    _say("ask: temporary server stopped")


def _next_steps(ctx: Context) -> list[tuple[str, str]]:
    question = SAMPLE_QUESTION.replace('"', '\\"')
    last = (
        (UPGRADE_HINT, "upgrade from offline mode to a local model")
        if ctx.choice is not None and ctx.choice.kind == "offline"
        else ("ragfabric config validate --check-providers", "make one live call per provider")
    )
    return [
        (f"cd {ctx.dir} && ragfabric serve", "run the API on http://127.0.0.1:8000"),
        ("ragfabric ingest ./my-docs --recursive", "add your own documents"),
        (
            f'ragfabric ask "{question}" --token "$RAGFABRIC_TOKEN"',
            "ask through the running server (token from POST /api/auth/login)",
        ),
        ("ragfabric doctor", "check config, database, migrations and providers"),
        last,
    ]


def quickstart(
    dir: Path = typer.Option(
        Path("."), "--dir", help="Directory to set up (created if missing).", file_okay=False
    ),
    docker: bool = typer.Option(
        False, "--docker", help="Use PostgreSQL and Redis from docker compose instead of SQLite."
    ),
    force: bool = typer.Option(
        False, "--force", help="Overwrite existing .env, ragfabric.yaml and docker-compose.yml."
    ),
    yes: bool = typer.Option(False, "--yes", help="Do not ask; take the default at every prompt."),
    model_check: bool = typer.Option(
        True,
        "--model-check/--no-model-check",
        help="Probe a local Ollama for models (2 s). Off: choose from API keys or offline.",
    ),
) -> None:
    """Set up a directory and get a first cited answer, with nothing installed.

    Writes .env and ragfabric.yaml, picks the best model available (Ollama,
    then an OpenAI or Anthropic key, then offline), creates a SQLite database
    (or PostgreSQL with --docker), migrates it, ingests three sample documents,
    and asks a sample question through a temporary server that is always
    stopped afterwards. Every step that is already done is skipped, so it is
    safe to run again. Nothing is downloaded and no secret is printed.

    Examples:

      ragfabric quickstart

      ragfabric quickstart --dir ./my-rag

      ragfabric quickstart --docker --yes

      ragfabric quickstart --force --no-model-check
    """
    dir = dir.expanduser().resolve()
    dir.mkdir(parents=True, exist_ok=True)
    ctx = Context(dir=dir, force=force, yes=yes, docker=docker, model_check=model_check)
    try:
        _step_config(ctx)
        _step_database(ctx)
        _step_model(ctx)
        _step_migrate(ctx)
        _step_ingest(ctx)
        _step_ask(ctx)
    except QuickstartError as exc:
        typer.echo(f"error: {mask_urls_in(str(exc))}", err=True)
        raise typer.Exit(1) from None
    render_next_steps(_next_steps(ctx))
