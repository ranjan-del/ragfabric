"""ASGI middleware: request ids and the access log line.

Written as plain ASGI rather than Starlette's ``BaseHTTPMiddleware``, which
re-wraps streaming responses and does not reliably carry context variables
into the endpoint. ``/api/ask`` streams Server-Sent Events, so that matters
here: the request id must still be bound while the stream is being produced.
"""

from __future__ import annotations

import json
import logging
import time

from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

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
                extra={"request_id": request_id},
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
                    "request_id": request_id,
                    "method": method,
                    "path": path,
                    "status": status,
                    "duration_ms": duration_ms,
                    "client": client[0] if client else None,
                },
            )
            reset_request_id(token)
