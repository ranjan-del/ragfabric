"""Structured logs: one JSON object per line, with the request id attached.

Every library RagFabric uses (uvicorn, SQLAlchemy, httpx, the provider SDKs)
already logs through the standard library, so a formatter on the root handler
is enough to make all of them structured. No ``structlog``: it would still
need a bridge for those libraries, and adds a dependency for nothing extra.

The request id lives in a context variable. The server's middleware binds it
for the life of a request (including a streamed response), the worker binds
``job:<id>`` for each job, and every log line written meanwhile, by any
logger, carries it.

Format and level come from ``ragfabric.yaml`` (``logging.format``,
``logging.level``) and can be overridden by ``RAGFABRIC_LOG_FORMAT`` and
``RAGFABRIC_LOG_LEVEL``, so a container can switch to JSON without editing a
mounted file.
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
import re
import sys
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import IO

_request_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "ragfabric_request_id", default=None
)

# 1 to 128 characters that cannot break a log line or a header: no spaces,
# quotes, braces or control characters, so a caller's id can never inject a
# newline or a fake JSON field into the log stream.
_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]{1,128}$")

FORMATS = ("text", "json")
LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")

# Attributes every LogRecord has. Anything else on a record came from
# ``extra=`` and is emitted as its own field.
_STANDARD_ATTRS = set(logging.LogRecord("x", logging.INFO, __file__, 1, "", (), None).__dict__) | {
    "message",
    "asctime",
    "taskName",
}

# The handler this module installed last (handlers it installs are also
# marked ``_ragfabric`` so configure_logging replaces them, never stacks).
_installed: logging.Handler | None = None


def _install_record_factory() -> None:
    """Stamp ``request_id`` on every LogRecord at creation, for every handler.

    A formatter alone would only label lines that pass through RagFabric's own
    handler. Stamping the record means any handler (a test's capture, a
    third-party shipper's) sees the id too. Because the record already has the
    attribute, passing ``request_id`` in ``extra=`` raises; bind it instead.
    """
    previous = logging.getLogRecordFactory()
    if getattr(previous, "_ragfabric", False):
        return

    def factory(*args, **kwargs) -> logging.LogRecord:
        record = previous(*args, **kwargs)
        record.request_id = _request_id.get()
        return record

    factory._ragfabric = True  # type: ignore[attr-defined]
    logging.setLogRecordFactory(factory)


_install_record_factory()


def valid_request_id(value: object) -> bool:
    return isinstance(value, str) and bool(_SAFE_ID.match(value))


def new_request_id() -> str:
    return uuid.uuid4().hex


def current_request_id() -> str | None:
    return _request_id.get()


def set_request_id(value: str | None) -> contextvars.Token:
    return _request_id.set(value)


def reset_request_id(token: contextvars.Token) -> None:
    _request_id.reset(token)


@contextmanager
def bind_request_id(value: str) -> Iterator[None]:
    token = _request_id.set(value)
    try:
        yield
    finally:
        _request_id.reset(token)


def _otel_ids() -> dict[str, str]:
    try:
        from opentelemetry import trace
    except ImportError:  # pragma: no cover - opentelemetry is a core dependency
        return {}
    ctx = trace.get_current_span().get_span_context()
    if not ctx.is_valid:
        return {}
    return {"trace_id": format(ctx.trace_id, "032x"), "span_id": format(ctx.span_id, "016x")}


def _jsonable(value: object) -> object:
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return str(value)
    return value


class JsonFormatter(logging.Formatter):
    """One JSON object per line: ts, level, logger, msg, request_id, extras, exc."""

    def format(self, record: logging.LogRecord) -> str:
        data: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, UTC)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        request_id = getattr(record, "request_id", None) or _request_id.get()
        if request_id is not None:
            data["request_id"] = request_id
        data.update(_otel_ids())
        for key, value in record.__dict__.items():
            if key not in _STANDARD_ATTRS and key not in data and key != "request_id":
                data[key] = _jsonable(value)
        if record.exc_info:
            data["exc"] = self.formatException(record.exc_info)
        elif record.exc_text:
            data["exc"] = record.exc_text
        return json.dumps(data, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    """Readable lines for a terminal, with ``[request_id]`` when there is one."""

    def __init__(self) -> None:
        super().__init__("%(asctime)s %(levelname)s %(name)s %(rid)s%(message)s")

    def format(self, record: logging.LogRecord) -> str:
        request_id = getattr(record, "request_id", None) or _request_id.get()
        record.rid = f"[{request_id}] " if request_id else ""
        return super().format(record)


class _CurrentStderrHandler(logging.StreamHandler):
    """Writes to whatever ``sys.stderr`` is at the moment of each record.

    A plain ``StreamHandler(sys.stderr)`` keeps the stream object it was given.
    If something later swaps ``sys.stderr`` (a test runner, a CLI runner, a
    daemon that redirects it) and closes the old one, every log line after
    that fails. Looking the stream up per record, as the standard library's own
    last-resort handler does, avoids it.
    """

    def __init__(self) -> None:
        super().__init__()

    @property
    def stream(self):  # type: ignore[override]
        return sys.stderr

    @stream.setter
    def stream(self, value) -> None:
        pass


def resolve(fmt: str, level: str) -> tuple[str, str]:
    """Apply the environment overrides and check both values."""
    fmt = os.environ.get("RAGFABRIC_LOG_FORMAT", "").strip().lower() or fmt
    level = os.environ.get("RAGFABRIC_LOG_LEVEL", "").strip().upper() or level.upper()
    if fmt not in FORMATS:
        raise ValueError(f"RAGFABRIC_LOG_FORMAT / logging.format must be one of {FORMATS}: {fmt!r}")
    if level not in LEVELS:
        raise ValueError(f"RAGFABRIC_LOG_LEVEL / logging.level must be one of {LEVELS}: {level!r}")
    return fmt, level


def configure_logging(
    fmt: str = "text", level: str = "INFO", *, stream: IO[str] | None = None
) -> tuple[str, str]:
    """Install one formatter on the root logger and return the (format, level) used.

    Safe to call more than once: the handler this module installed last time is
    replaced, never stacked.
    """
    global _installed
    fmt, level = resolve(fmt, level)
    root = logging.getLogger()
    for existing in list(root.handlers):
        if getattr(existing, "_ragfabric", False):
            root.removeHandler(existing)
    handler = logging.StreamHandler(stream) if stream is not None else _CurrentStderrHandler()
    handler.setFormatter(JsonFormatter() if fmt == "json" else TextFormatter())
    handler._ragfabric = True  # type: ignore[attr-defined]
    root.addHandler(handler)
    root.setLevel(level)
    _installed = handler
    # uvicorn installs its own handlers on these loggers and stops them
    # propagating, which would bypass the formatter above. Hand them to the
    # root handler instead. uvicorn's access log is silenced outright: the
    # server's RequestContextMiddleware writes one access line per request,
    # with the request id, so uvicorn's would only be a second, id-less copy.
    for name in ("uvicorn", "uvicorn.error"):
        adopted = logging.getLogger(name)
        adopted.handlers.clear()
        adopted.propagate = True
    access = logging.getLogger("uvicorn.access")
    access.handlers.clear()
    access.propagate = False
    return fmt, level
