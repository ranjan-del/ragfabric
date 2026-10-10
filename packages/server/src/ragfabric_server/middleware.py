"""ASGI middleware: request ids, the access log line and the upload size bound.

Written as plain ASGI rather than Starlette's ``BaseHTTPMiddleware``, which
re-wraps streaming responses and does not reliably carry context variables
into the endpoint. ``/api/ask`` streams Server-Sent Events, so that matters
here: the request id must still be bound while the stream is being produced.
"""

from __future__ import annotations

import json
import logging
import time

from fastapi import HTTPException
from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from ragfabric_core.runtime import get_config
from ragfabric_core.telemetry.logs import (
    new_request_id,
    reset_request_id,
    set_request_id,
    valid_request_id,
)

REQUEST_ID_HEADER = "X-Request-ID"

access_log = logging.getLogger("ragfabric.access")
error_log = logging.getLogger("ragfabric.server")


class RequestContextMiddleware:
    """Bind a request id for the whole request, echo it, log one access line.

    A caller's ``X-Request-ID`` is kept when it is safe to log (see
    ``valid_request_id``); otherwise a fresh one is generated. An exception
    nothing else handled becomes a JSON 500 carrying the id, with the
    traceback logged under that id and never sent to the caller.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        incoming = Headers(scope=scope).get(REQUEST_ID_HEADER)
        request_id = incoming if valid_request_id(incoming) else new_request_id()
        scope.setdefault("state", {})["request_id"] = request_id
        token = set_request_id(request_id)
        started = time.perf_counter()
        status = 500
        response_started = False

        async def send_with_id(message: Message) -> None:
            nonlocal status, response_started
            if message["type"] == "http.response.start":
                response_started = True
                status = message["status"]
                MutableHeaders(scope=message)[REQUEST_ID_HEADER] = request_id
            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        except Exception:
            error_log.exception(
                "unhandled error on %s %s",
                scope.get("method", ""),
                scope.get("path", ""),
            )
            if response_started:
                raise
            status = 500
            body = json.dumps(
                {"detail": "Internal server error", "request_id": request_id}
            ).encode()
            await send(
                {
                    "type": "http.response.start",
                    "status": 500,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"content-length", str(len(body)).encode()),
                        (REQUEST_ID_HEADER.lower().encode(), request_id.encode()),
                    ],
                }
            )
            await send({"type": "http.response.body", "body": body})
        finally:
            duration_ms = int((time.perf_counter() - started) * 1000)
            client = scope.get("client")
            method, path = scope.get("method", ""), scope.get("path", "")
            access_log.info(
                "%s %s %s %dms",
                method,
                path,
                status,
                duration_ms,
                extra={
                    "method": method,
                    "path": path,
                    "status": status,
                    "duration_ms": duration_ms,
                    "client": client[0] if client else None,
                },
            )
            reset_request_id(token)


# Paths whose body is an uploaded file, and the multipart framing allowance on
# top of limits.max_upload_mb (boundaries, part headers, the other form fields).
UPLOAD_PATHS = frozenset({"/api/documents/upload"})
MULTIPART_SLACK_BYTES = 1024 * 1024


class _BodyTooLarge(HTTPException):
    """Raised from ``receive`` mid-parse. An HTTPException, so FastAPI's form
    parsing re-raises it as it is rather than turning it into a 400."""

    def __init__(self, limit_mb: int) -> None:
        super().__init__(
            status_code=413,
            detail=f"Upload is over the configured limit of {limit_mb} MB.",
        )


class UploadSizeLimitMiddleware:
    """Bound an upload's size on the wire, before the multipart parser spools it.

    Without this, the whole body is parsed and written to a temporary file
    before the route can look at its size, so a 10 GB upload is written to disk
    and only then refused. A declared ``Content-Length`` over the limit is
    refused at once without calling the app; a body without one (chunked) is
    counted as it arrives and stopped once it passes the limit.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("path") not in UPLOAD_PATHS:
            await self.app(scope, receive, send)
            return

        limit_mb = get_config().limits.max_upload_mb
        allowed = limit_mb * 1024 * 1024 + MULTIPART_SLACK_BYTES
        declared = Headers(scope=scope).get("content-length")
        if declared is not None and declared.isdigit() and int(declared) > allowed:
            body = json.dumps(
                {"detail": f"Upload is over the configured limit of {limit_mb} MB."}
            ).encode()
            await send(
                {
                    "type": "http.response.start",
                    "status": 413,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"content-length", str(len(body)).encode()),
                        (b"connection", b"close"),
                    ],
                }
            )
            await send({"type": "http.response.body", "body": body})
            return

        received = 0

        async def counting_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > allowed:
                    raise _BodyTooLarge(limit_mb)
            return message

        await self.app(scope, counting_receive, send)
