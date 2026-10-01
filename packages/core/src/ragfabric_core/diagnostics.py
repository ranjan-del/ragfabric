"""Environment checks behind ``ragfabric doctor``.

Each check reports only what it actually examined (ADR 0004): a check that did
not run is ``skip`` with the reason, never ``pass``. Every detail that could
carry a URL goes through ``mask_url`` first, so no password is ever printed.
"""

from __future__ import annotations

import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import ValidationError

from ragfabric_core.config_file import RagFabricConfig, load_config, resolve_config_path
from ragfabric_core.db import migrate
from ragfabric_core.providers.base import Message, ProviderError
from ragfabric_core.providers.registry import build_embedding_provider, build_llm_provider

_URL = re.compile(r"\b[a-z][a-z0-9+.-]*://\S+", re.IGNORECASE)
SERVER_TIMEOUT_SECONDS = 2


def mask_url(url: str) -> str:
    """Replace the password in a URL with ``***``; anything else comes back unchanged.

    The authority is everything between ``://`` and the next ``/``, and the
    password runs to the LAST ``@`` in it, so a password that itself contains
    ``@`` is masked whole.
    """
    marker = url.find("://")
    if marker == -1:
        return url
    start = marker + 3
    slash = url.find("/", start)
    end = len(url) if slash == -1 else slash
    authority = url[start:end]
    at = authority.rfind("@")
    if at == -1:
        return url
    userinfo = authority[:at]
    colon = userinfo.find(":")
    if colon == -1:
        return url
    return url[:start] + userinfo[: colon + 1] + "***" + authority[at:] + url[end:]


def mask_urls_in(text: str) -> str:
    """Mask the password of every URL found in ``text``."""
    return _URL.sub(lambda m: mask_url(m.group(0)), text)


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: Literal["pass", "warn", "fail", "skip"]
    detail: str
    fix: str | None = None


class MigrationsBehind(Exception):
    def __init__(self, current: str | None, head: str) -> None:
        super().__init__(f"database is at {current or 'no revision'}, code expects {head}")
        self.current = current
        self.head = head


def check_python() -> CheckResult:
    v = sys.version_info
    version = f"{v.major}.{v.minor}.{v.micro}"
    if (v.major, v.minor) == (3, 13):
        return CheckResult("python", "pass", f"Python {version}")
    return CheckResult("python", "fail", f"Python {version}", "use Python 3.13")


def check_config(path: Path | None) -> CheckResult:
    resolved = resolve_config_path(path)
    if resolved is None:
        return CheckResult(
            "config",
            "warn",
            "no ragfabric.yaml found, using defaults",
            "ragfabric quickstart, or ragfabric init",
        )
    try:
        load_config(path)
    except ValidationError as exc:
        errors = exc.errors()
        first = errors[0] if errors else {"loc": (), "msg": str(exc)}
        loc = ".".join(str(p) for p in first["loc"])
        detail = f"{loc}: {first['msg']}" if loc else str(first["msg"])
        return CheckResult(
            "config", "fail", f"{resolved} is invalid, {detail}", "fix the key above"
        )
    except (OSError, ValueError) as exc:
        return CheckResult("config", "fail", mask_urls_in(f"cannot read {resolved}: {exc}"), None)
    return CheckResult("config", "pass", f"{resolved} is valid")


def check_database(db_url: str) -> CheckResult:
    from sqlalchemy import create_engine, text

    shown = mask_urls_in(db_url)
    kwargs = {} if db_url.startswith("sqlite") else {"connect_args": {"connect_timeout": 3}}
    engine = None
    try:
        engine = create_engine(db_url, **kwargs)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception:  # noqa: BLE001 - any failure to connect is the finding
        return CheckResult(
            "database",
            "fail",
            f"cannot reach {shown}",
            "ragfabric quickstart for SQLite, or docker compose up -d postgres",
        )
    finally:
        if engine is not None:
            engine.dispose()
    return CheckResult("database", "pass", f"connected to {shown}")


def check_migrations(db_url: str) -> CheckResult:
    try:
        current = migrate.current_revision(db_url)
        head = migrate.head_revision()
    except Exception:  # noqa: BLE001 - an unreachable database is reported, not raised
        return CheckResult(
            "migrations",
            "fail",
            f"cannot read the schema revision from {mask_urls_in(db_url)}",
            None,
        )
    if current == head:
        return CheckResult("migrations", "pass", f"at head ({head})")
    return CheckResult(
        "migrations",
        "fail",
        f"database is at {current or 'no revision'}, code expects {head}",
        "ragfabric db upgrade",
    )


def check_llm(cfg: RagFabricConfig, *, network: bool) -> CheckResult:
    if not network:
        return CheckResult("llm", "skip", "not checked: network checks are off")
    if cfg.llm.provider == "offline":
        return CheckResult(
            "llm",
            "warn",
            "offline mode: answers are extractive",
            "install Ollama, or set OPENAI_API_KEY or ANTHROPIC_API_KEY, "
            "then ragfabric quickstart --force",
        )
    try:
        llm = build_llm_provider(cfg.llm)
        llm.complete([Message(role="user", content="ping")], max_tokens=1)
    except ProviderError as exc:
        return CheckResult(
            "llm", "fail", mask_urls_in(str(exc)), "check the provider settings in ragfabric.yaml"
        )
    return CheckResult("llm", "pass", f"{cfg.llm.provider} answered one completion")


def check_embeddings(cfg: RagFabricConfig, *, network: bool) -> CheckResult:
    if not network:
        return CheckResult("embeddings", "skip", "not checked: network checks are off")
    if cfg.embeddings.provider == "offline":
        return CheckResult(
            "embeddings",
            "warn",
            "offline mode: hashing embeddings, not semantic",
            "install Ollama, or set OPENAI_API_KEY, then ragfabric quickstart --force",
        )
    try:
        provider = build_embedding_provider(cfg.embeddings)
        vectors = provider.embed(["ping"]).vectors
    except ProviderError as exc:
        return CheckResult(
            "embeddings",
            "fail",
            mask_urls_in(str(exc)),
            "check the provider settings in ragfabric.yaml",
        )
    got = len(vectors[0]) if vectors else 0
    if got != cfg.embeddings.dim:
        return CheckResult(
            "embeddings",
            "fail",
            f"the model returned {got} dimensions, embeddings.dim is {cfg.embeddings.dim}",
            f"set embeddings.dim to {got} in ragfabric.yaml",
        )
    return CheckResult("embeddings", "pass", f"{cfg.embeddings.provider} returned {got} dimensions")


def check_graph(cfg: RagFabricConfig) -> CheckResult:
    if not cfg.graph_store.enabled:
        return CheckResult("graph", "skip", "not checked: graph_store.enabled is false")
    return CheckResult("graph", "pass", f"graph_store enabled in config ({cfg.graph_store.kind})")


def check_server(url: str) -> CheckResult:
    target = url.rstrip("/") + "/health"
    try:
        with urllib.request.urlopen(target, timeout=SERVER_TIMEOUT_SECONDS) as response:  # noqa: S310
            if 200 <= response.status < 300:
                return CheckResult("server", "pass", f"{mask_urls_in(url)} is healthy")
            status = response.status
    except urllib.error.HTTPError as exc:
        status = exc.code
    except (OSError, ValueError):
        return CheckResult(
            "server", "warn", f"no server answered at {mask_urls_in(url)}", "ragfabric serve"
        )
    return CheckResult(
        "server",
        "warn",
        f"{mask_urls_in(url)} answered HTTP {status} on /health",
        "ragfabric serve",
    )


def run_all(
    *, config_path: Path | None, db_url: str, server_url: str, network: bool
) -> list[CheckResult]:
    results = [check_python(), check_config(config_path)]
    try:
        cfg = load_config(config_path)
    except Exception:  # noqa: BLE001 - the config check already reported why
        cfg = None
    results.append(check_database(db_url))
    results.append(check_migrations(db_url))
    if cfg is None:
        reason = "not checked: the configuration is unreadable"
        results += [CheckResult(name, "skip", reason) for name in ("llm", "embeddings", "graph")]
    else:
        results.append(check_llm(cfg, network=network))
        results.append(check_embeddings(cfg, network=network))
        results.append(check_graph(cfg))
    results.append(
        check_server(server_url)
        if network
        else CheckResult("server", "skip", "not checked: network checks are off")
    )
    return results
