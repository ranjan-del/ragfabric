"""Turn the failures a user can fix into a problem line and a fix line."""

from __future__ import annotations

import errno
import re
from dataclasses import dataclass

import httpx
import typer
from pydantic import ValidationError
from sqlalchemy.exc import OperationalError
from typer._click.exceptions import ClickException

from ragfabric_cli.ui.console import mask_url, mask_urls_in
from ragfabric_core.providers.base import ProviderError
from ragfabric_sdk.errors import AuthError, RagFabricError

# Set by the --debug global option on the app callback.
DEBUG = False

_URL = re.compile(r"\b[a-z+]+://\S+")


@dataclass(frozen=True)
class FriendlyError:
    problem: str
    fix: str | None = None


def _migrations_behind_type() -> type | None:
    try:
        from ragfabric_core.diagnostics import MigrationsBehind
    except ImportError:
        return None
    return MigrationsBehind


def friendly_error(exc: BaseException) -> FriendlyError | None:
    """Map an exception to a FriendlyError, or None when it is not one we explain."""
    if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout)):
        try:
            request = exc.request
            url = f"{request.url.scheme}://{request.url.netloc.decode()}"
        except RuntimeError:
            url = "the configured URL"
        return FriendlyError(f"No RagFabric server at {url}", "ragfabric serve")
    if isinstance(exc, AuthError):
        return FriendlyError(
            "The server refused the credentials",
            "pass --token or --api-key, or set RAGFABRIC_TOKEN",
        )
    if isinstance(exc, RagFabricError):
        return FriendlyError(f"The server returned an error: {mask_urls_in(str(exc))}", None)
    if isinstance(exc, OperationalError):
        found = _URL.search(mask_urls_in(str(exc)))
        where = mask_url(found.group(0)) if found else "the configured database"
        return FriendlyError(
            f"Cannot reach the database at {where}",
            "ragfabric quickstart for SQLite, or docker compose up -d postgres",
        )
    behind = _migrations_behind_type()
    if behind is not None and isinstance(exc, behind):
        return FriendlyError("The database schema is behind", "ragfabric db upgrade")
    if isinstance(exc, ValidationError):
        errors = exc.errors()
        if errors:
            first = errors[0]
            loc = ".".join(str(p) for p in first["loc"])
            detail = f"{loc}: {first['msg']}" if loc else first["msg"]
        else:
            detail = str(exc)
        return FriendlyError(
            f"ragfabric.yaml is invalid: {mask_urls_in(detail)}", "ragfabric config validate"
        )
    if isinstance(exc, ProviderError):
        # ProviderError carries only provider and message, no status code, so the
        # 401 and 429 rows can only match on the message text.
        message = mask_urls_in(str(exc))
        provider = exc.provider
        if "401" in message or "invalid_api_key" in message:
            return FriendlyError(f"{provider} rejected the API key", "check the key in .env")
        if "429" in message or "insufficient_quota" in message:
            return FriendlyError(f"{provider} quota is used up or rate limited", None)
        if "Connection" in message:
            fix = "ollama serve" if provider == "ollama" else "check the provider URL"
            return FriendlyError(f"Cannot reach {provider}", fix)
        return FriendlyError(f"{provider}: {message}", None)
    return None


def report(exc: BaseException) -> None:
    """Print the friendly form of exc (or the unexpected form) to stderr."""
    fe = friendly_error(exc)
    if fe is None:
        typer.echo(f"Unexpected error: {type(exc).__name__}: {mask_urls_in(str(exc))}", err=True)
        typer.echo("rerun with --debug for details", err=True)
        return
    typer.echo(f"Error: {fe.problem}", err=True)
    if fe.fix:
        typer.echo(f"Fix: {fe.fix}", err=True)


class FriendlyGroup(typer.core.TyperGroup):
    """Top level group that turns stray exceptions into friendly errors.

    Living on the group (not only in run_app) means CliRunner and the real
    entry point behave the same.
    """

    def invoke(self, ctx: typer.Context):  # type: ignore[override]
        try:
            return super().invoke(ctx)
        except (ClickException, typer.Exit, typer.Abort):
            raise
        except Exception as exc:
            # Piping into `head` closes stdout early; that is not an error.
            if isinstance(exc, BrokenPipeError) or getattr(exc, "errno", None) == errno.EPIPE:
                raise typer.Exit(1) from None
            if DEBUG:
                raise
            report(exc)
            raise typer.Exit(1) from None


def run_app(app: typer.Typer) -> None:
    """Run the app. Errors are handled by FriendlyGroup; --debug re-raises."""
    app()
