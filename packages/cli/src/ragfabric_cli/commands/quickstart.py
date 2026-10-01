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

import importlib.util
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import httpx
import typer

from ragfabric_cli.commands.ingest import ingest_files
from ragfabric_cli.envfile import read_env_file
from ragfabric_cli.templates import SAMPLE_QUESTION, sample_paths, template_path
from ragfabric_cli.ui import console
from ragfabric_cli.ui.console import mask_urls_in
from ragfabric_cli.ui.panels import render_answer, render_next_steps
from ragfabric_core.diagnostics import free_port, port_in_use

OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"
OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_CHAT_MODEL = "llama3.2:3b"
OLLAMA_EMBED_MODEL = "nomic-embed-text"
# Migration 0004 pins the pgvector column at 768, the nomic-embed-text width.
PINNED_DIM = 768
HEALTH_TIMEOUT_S = 30.0
STOP_TIMEOUT_S = 5.0
DB_WAIT_S = 60.0

SERVE_URL = "http://127.0.0.1:8000"
API_KEY_NAME = "quickstart"


OPENAI_INSTALL = "pip install 'ragfabric[openai]'"
ANTHROPIC_INSTALL = "pip install 'ragfabric[anthropic]'"


def installed(package: str) -> bool:
    """True when the optional client ``package`` can be imported. Nothing is imported."""
    return importlib.util.find_spec(package) is not None


def upgrade_hint(dir: Path | None) -> str:
    """The command that moves an offline setup to a local model.

    Never --force: that would re-copy .env and lose the API key, JWT_SECRET and
    any Docker DATABASE_URL. Moving ragfabric.yaml aside makes quickstart
    choose the model again and leaves everything else as it is. Ollama is
    served through the openai client package, so the hint installs it first
    while it is missing. ``dir`` None leaves out the ``cd``, for a list of
    steps that already changed into the directory.
    """
    install = f"{OPENAI_INSTALL} && " if not installed("openai") else ""
    cd = f"cd {shlex.quote(str(dir))} && " if dir is not None else ""
    return (
        f"{install}ollama pull {OLLAMA_CHAT_MODEL} && ollama pull {OLLAMA_EMBED_MODEL} && "
        f"{cd}mv ragfabric.yaml ragfabric.yaml.bak && ragfabric quickstart && "
        "ragfabric reindex --yes"
    )


class QuickstartError(Exception):
    """A step failed in a way the user can act on. The message says how."""


class PortInUse(QuickstartError):
    """The temporary server could not bind its port: something took it after free_port."""


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
    key_in_env: bool = False
    serve_url: str | None = None


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
            if chat and embed and not installed("openai"):
                model = OLLAMA_CHAT_MODEL if OLLAMA_CHAT_MODEL in chat else chat[0]
                notes.append(
                    f"Ollama is running with {model} and {OLLAMA_EMBED_MODEL}, but its client "
                    f"package (openai) is not installed (run: {OPENAI_INSTALL})"
                )
            elif chat and embed:
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
            else:
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

    if env.get("OPENAI_API_KEY") and not installed("openai"):
        notes.append(
            "OPENAI_API_KEY is set, but its client package (openai) is not installed "
            f"(run: {OPENAI_INSTALL})"
        )
    elif env.get("OPENAI_API_KEY"):
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
    if env.get("ANTHROPIC_API_KEY") and not installed("anthropic"):
        notes.append(
            "ANTHROPIC_API_KEY is set, but its client package (anthropic) is not installed "
            f"(run: {ANTHROPIC_INSTALL})"
        )
    elif env.get("ANTHROPIC_API_KEY"):
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
                (
                    "no model is usable"
                    if env.get("OPENAI_API_KEY") or env.get("ANTHROPIC_API_KEY")
                    else "no OPENAI_API_KEY or ANTHROPIC_API_KEY is set"
                )
                + ", so offline mode: answers are "
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
        src = template_path(template)
        if dst.exists() and not ctx.force:
            if target == "ragfabric.yaml" and dst.read_bytes() == src.read_bytes():
                # Byte for byte the packaged template: an earlier run copied it and
                # stopped before the model step patched it. Nobody edited it.
                ctx.config_written = True
                _say("config: ragfabric.yaml is the unpatched template, patching it")
                continue
            _done("config", f"{target} exists, keeping it (use --force to overwrite)")
            continue
        if target == ".env":
            _write_private(dst, src.read_bytes())
        else:
            shutil.copyfile(src, dst)
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
        _say(f"  upgrade later with: {upgrade_hint(ctx.dir)}")


def _read_env_value(path: Path, key: str) -> str | None:
    return read_env_file(path).get(key) or None


def _write_private(path: Path, data: bytes) -> None:
    """Write ``path`` readable and writable by its owner only (0600): .env holds secrets."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)
    os.chmod(path, 0o600)  # O_CREAT's mode does not apply to a file that already existed


def _set_env_value(path: Path, key: str, value: str) -> None:
    lines = path.read_text().splitlines(keepends=True) if path.is_file() else []
    for i, line in enumerate(lines):
        if line.partition("=")[0].strip() == key:
            lines[i] = f"{key}={value}\n"
            break
    else:
        lines.append(f"{key}={value}\n")
    _write_private(path, "".join(lines).encode("utf-8"))


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
    """Record the database, and make .env name it.

    A .env this run wrote is simply updated. A kept .env whose DATABASE_URL
    names another database (an earlier SQLite run, then --docker) gets that
    one key updated after a yes at the prompt (--yes means yes); everything
    else in it stays. Declining stops before anything is migrated.
    """
    ctx.db_url = url
    env_path = ctx.dir / ".env"
    if ctx.env_written:
        _set_env_value(env_path, "DATABASE_URL", url)
        _say(f"database: {what} at {url}")
        return
    current = _read_env_value(env_path, "DATABASE_URL")
    if current is not None and _absolute_sqlite(current, ctx.dir) == url:
        _done("database", f"{what} at {url}")
        return
    if not _looks_like_ragfabric_env(read_env_file(env_path)):
        raise QuickstartError(
            f"the .env in {ctx.dir} is not RagFabric's (DATABASE_URL "
            f"{mask_urls_in(current or 'not set')})\nfix: ragfabric quickstart --dir ./ragfabric"
        )
    shown = mask_urls_in(current) if current else "no database"
    if not ctx.yes and not typer.confirm(
        f".env names {shown}. Update DATABASE_URL in .env to {mask_urls_in(url)}?",
        default=True,
    ):
        raise QuickstartError(
            f".env still names {shown}, so nothing was migrated; "
            f"set DATABASE_URL in .env to {mask_urls_in(url)} to use {what}"
        )
    _set_env_value(env_path, "DATABASE_URL", url)
    _say(f"database: {what} at {url}; updated DATABASE_URL in .env (was {shown})")


_RAGFABRIC_ENV_KEYS = ("JWT_SECRET", "FIRST_ADMIN_EMAIL")


def _looks_like_ragfabric_env(values: Mapping[str, str]) -> bool:
    """True when a .env carries a key only RagFabric's template writes."""
    return any(k in values for k in _RAGFABRIC_ENV_KEYS) or any(
        k.startswith("RAGFABRIC_") for k in values
    )


def _sqlite_inside(url: str, dir: Path) -> bool:
    """True for a SQLite file URL whose file is inside ``dir``."""
    prefix = "sqlite:///"
    if not url.startswith(prefix):
        return False
    path = url[len(prefix) :]
    if not path or path == ":memory:":
        return False
    return Path(path).resolve().is_relative_to(dir.resolve())


def _check_kept_database(ctx: Context, url: str) -> None:
    """Refuse to migrate a database that a kept .env names but RagFabric does not own.

    A .env without any RagFabric key belongs to some other program: its
    DATABASE_URL is that program's database, so nothing is migrated into it.
    A RagFabric .env naming anything but a SQLite file in this directory (or
    the compose PostgreSQL an earlier --docker run wrote) is migrated only
    when the user says yes at the prompt; --yes never says yes to it.
    """
    values = read_env_file(ctx.dir / ".env")
    fix = "fix: ragfabric quickstart --dir ./ragfabric"
    if not _looks_like_ragfabric_env(values):
        raise QuickstartError(
            f"the .env in {ctx.dir} is not RagFabric's (DATABASE_URL {mask_urls_in(url)})\n{fix}"
        )
    if _sqlite_inside(url, ctx.dir) or url == _docker_url(ctx.dir):
        return
    if not ctx.yes and typer.confirm(
        f"Migrate and ingest into the database at {mask_urls_in(url)} named in .env?",
        default=False,
    ):
        return
    raise QuickstartError(
        f"the .env in {ctx.dir} names a database outside it (DATABASE_URL {mask_urls_in(url)}), "
        f"so nothing was migrated\n{fix}"
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
        url = _absolute_sqlite(existing, ctx.dir)
        _check_kept_database(ctx, url)
        ctx.db_url = url
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


@contextmanager
def _quiet_alembic() -> Iterator[None]:
    """Keep alembic at WARNING while migrating, so its INFO lines stay out of the output.

    Setting the logger level alone does not hold: the packaged env.py runs
    fileConfig, which puts the alembic logger back to INFO. Disabling INFO
    for the duration does hold, and is undone afterwards.
    """
    import logging

    previous = logging.root.manager.disable
    logging.getLogger("alembic").setLevel(logging.WARNING)
    logging.disable(logging.INFO)
    try:
        yield
    finally:
        logging.disable(previous)


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
    with _spinner("migrate: applying migrations"), _quiet_alembic():
        migrate.upgrade(ctx.db_url)
    _say(f"migrate: database upgraded to {head}")


def _present_filenames(names: list[str]) -> set[str]:
    """The sample filenames already ingested successfully. A failed one is not present."""
    from ragfabric_cli.commands.common import session
    from ragfabric_core.models.document import Document

    with session() as db:
        rows = (
            db.query(Document.filename)
            .filter(Document.filename.in_(names), Document.status == "ready")
            .all()
        )
    return {row[0] for row in rows}


def _step_ingest(ctx: Context) -> None:
    samples = sample_paths()
    with _bound_runtime(ctx):
        present = _present_filenames([p.name for p in samples])
        missing = [p for p in samples if p.name not in present]
        if not missing:
            _done("ingest", f"{len(samples)} sample documents present")
            return

        ingested, failed = ingest_files(missing, collection=None, owner=None, quiet=False)
    _say(f"ingest: {ingested} ingested, {failed} failed")
    if failed:
        raise QuickstartError("some sample documents failed to ingest; see the lines above")


# Ask ------------------------------------------------------------------------


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
                    tail = _log_tail(log)
                    error = (
                        PortInUse if "address already in use" in tail.lower() else QuickstartError
                    )
                    raise error(
                        f"the temporary server exited with code {proc.returncode}:\n" + tail
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


def _start_server(ctx: Context, stack: ExitStack) -> str:
    """Enter a temporary server on a free port, retrying once if the port was taken."""
    for attempt in (1, 2):
        port = free_port()
        _say(f"ask: starting a temporary server on 127.0.0.1:{port}")
        try:
            return stack.enter_context(temporary_server(ctx.dir, port, env=_child_env(ctx)))
        except PortInUse:
            if attempt == 2:
                raise
            _say(f"ask: port {port} was taken before the server bound it, trying another")
    raise AssertionError("unreachable")


def _create_api_key(ctx: Context, email: str) -> str:
    """Create an API key for the bootstrap admin, the way ``ragfabric keys create`` does."""
    from ragfabric_cli.commands.common import session
    from ragfabric_core.auth.api_keys import create_api_key
    from ragfabric_core.models.access import ApiKey
    from ragfabric_core.models.user import User

    with _bound_runtime(ctx), session() as db:
        user = db.query(User).filter(User.email == email.lower()).first()
        if user is None:
            raise QuickstartError(f"the bootstrap admin {email} was not created by the server")
        # Retire the keys earlier runs created (a --force rerun writes a new .env,
        # so the old key would otherwise stay valid with nothing holding it).
        retired = (
            db.query(ApiKey)
            .filter(
                ApiKey.principal_user_id == user.id,
                ApiKey.name == API_KEY_NAME,
                ApiKey.is_active.is_(True),
            )
            .all()
        )
        for old in retired:
            old.is_active = False
        _, plaintext = create_api_key(db, name=API_KEY_NAME, user_id=user.id)
        db.commit()
    return plaintext


def _key_is_active(ctx: Context, key: str) -> bool:
    """True when ``key`` is an active API key in the database this run uses."""
    from ragfabric_cli.commands.common import session
    from ragfabric_core.auth.api_keys import verify_api_key

    with _bound_runtime(ctx), session() as db:
        return verify_api_key(db, key) is not None


def _key_command(ctx: Context, email: str) -> str:
    return (
        f"cd {shlex.quote(str(ctx.dir))} && export RAGFABRIC_API_KEY="
        f'"$(ragfabric keys create --name cli --user {shlex.quote(email)} | tail -n 1)"'
    )


def _choose_serve_url() -> str:
    """http://127.0.0.1:8000, or a free port when another program holds 8000."""
    if port_in_use("127.0.0.1", 8000):
        return f"http://127.0.0.1:{free_port()}"
    return SERVE_URL


def _serve_port(ctx: Context) -> int | None:
    """The port ``ragfabric serve`` must use for .env's RAGFABRIC_URL, or None for 8000."""
    from urllib.parse import urlsplit

    url = ctx.serve_url or _read_env_value(ctx.dir / ".env", "RAGFABRIC_URL")
    if not url:
        return None
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError:
        return None
    if parts.hostname not in ("127.0.0.1", "localhost") or port in (None, 8000):
        return None
    return port


def _credentials(ctx: Context, url: str, email: str, password: str) -> dict[str, str]:
    """Client credentials for the sample question. The key itself is never printed."""
    env_path = ctx.dir / ".env"
    existing = _read_env_value(env_path, "RAGFABRIC_API_KEY")
    if existing and _key_is_active(ctx, existing):
        ctx.key_in_env = True
        _done("api key", "RAGFABRIC_API_KEY is in .env")
        return {"api_key": existing}
    if existing:
        # Written for another database (an earlier SQLite run, before --docker):
        # this database has never heard of it. Replace quickstart's line.
        key = _create_api_key(ctx, email)
        _set_env_value(env_path, "RAGFABRIC_API_KEY", key)
        ctx.key_in_env = True
        _say(
            "api key: the RAGFABRIC_API_KEY in .env is not active in this database; created "
            f"key {API_KEY_NAME!r} for {email} and replaced it in .env"
        )
        return {"api_key": key}
    if ctx.env_written:
        key = _create_api_key(ctx, email)
        _set_env_value(env_path, "RAGFABRIC_API_KEY", key)
        serve_url = _choose_serve_url()
        _set_env_value(env_path, "RAGFABRIC_URL", serve_url)
        ctx.key_in_env = True
        ctx.serve_url = serve_url
        _say(
            f"api key: created key {API_KEY_NAME!r} for {email}; saved to .env as "
            f"RAGFABRIC_API_KEY, with RAGFABRIC_URL={serve_url}"
        )
        if serve_url != SERVE_URL:
            _say(
                "api key: 127.0.0.1:8000 is in use by another program, so .env names "
                f"{serve_url}; start the server with: ragfabric serve --port "
                f"{serve_url.rsplit(':', 1)[1]}"
            )
        return {"api_key": key}
    res = httpx.post(
        f"{url}/api/auth/login", data={"username": email, "password": password}, timeout=10
    )
    if res.status_code != 200:
        raise QuickstartError(
            "could not sign in as the bootstrap admin from .env "
            f"(HTTP {res.status_code}); check FIRST_ADMIN_EMAIL and FIRST_ADMIN_PASSWORD"
        )
    _say(
        "api key: .env was kept, so no key is written to it. Create one and export it with: "
        + _key_command(ctx, email)
    )
    return {"token": res.json()["access_token"]}


def _step_ask(ctx: Context) -> None:
    from ragfabric_sdk import Client

    email, password = _admin_credentials(ctx)
    with ExitStack() as stack:
        url = _start_server(ctx, stack)
        credentials = _credentials(ctx, url, email, password)
        _say(f"ask: {SAMPLE_QUESTION}")
        with Client(url, timeout=120, **credentials) as client:
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


def _is_offline(ctx: Context) -> bool:
    if ctx.choice is not None:
        return ctx.choice.kind == "offline"
    try:
        from ragfabric_core.config_file import load_config

        return load_config(ctx.dir / "ragfabric.yaml").llm.provider == "offline"
    except Exception:  # noqa: BLE001 - an unreadable config just means no upgrade hint
        return False


def _next_steps(ctx: Context) -> list[tuple[str, str]]:
    """What to run next, copyable as written.

    One ``cd`` step first when --dir is not the current directory, then bare
    commands: a ``cd <dir> &&`` prefix on every line hid the command itself.
    """
    port = _serve_port(ctx)
    question = SAMPLE_QUESTION.replace('"', '\\"')
    ask_why = (
        "ask the running server; RAGFABRIC_URL and RAGFABRIC_API_KEY come from .env"
        if ctx.key_in_env
        else "ask the running server, after exporting RAGFABRIC_API_KEY as shown above"
    )
    last = (
        (upgrade_hint(None), "upgrade from offline mode to a local model")
        if _is_offline(ctx)
        else ("ragfabric config validate --check-providers", "make one live call per provider")
    )
    steps: list[tuple[str, str]] = []
    if ctx.dir.resolve() != Path.cwd().resolve():
        steps.append((f"cd {shlex.quote(str(ctx.dir))}", "the directory quickstart set up"))
    serve_why = (
        f"run the API on http://127.0.0.1:{port or 8000}; "
        "leave running; use a second terminal for the rest"
    )
    if ctx.docker and not ctx.db_url.startswith("sqlite"):
        serve_why += " (docker compose down stops PostgreSQL and Redis when you are done)"
    return [
        *steps,
        ("ragfabric serve" + (f" --port {port}" if port else ""), serve_why),
        ("ragfabric ingest ./my-docs --recursive", "add your own documents"),
        (f'ragfabric ask "{question}"', ask_why),
        ("ragfabric doctor", "check config, database, migrations and providers"),
        ("ragfabric strategies", "see the retrieval strategies and when each is best"),
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

    \b
    Examples:
      ragfabric quickstart
      ragfabric quickstart --dir ./my-rag
      ragfabric quickstart --docker --yes
      ragfabric quickstart --no-model-check
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
