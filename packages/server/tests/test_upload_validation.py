"""Upload validation over HTTP and the on-the-wire size limit (Phase 10, Task 3)."""

from __future__ import annotations

import pytest

from ragfabric_core import runtime
from ragfabric_core.testing.fixtures import make_docx, make_pdf


@pytest.fixture()
def limits():
    cfg = runtime.get_config().limits
    saved = cfg.max_upload_mb
    yield cfg
    cfg.max_upload_mb = saved


def _upload(client, headers, name, data, content_type="application/octet-stream"):
    return client.post(
        "/api/documents/upload", files={"file": (name, data, content_type)}, headers=headers
    )


def test_a_genuine_pdf_is_accepted(client, admin_headers):
    res = _upload(client, admin_headers, "a.pdf", make_pdf(["annual leave is ten days"]))
    assert res.status_code == 201, res.text


def test_a_fake_pdf_is_415_and_nothing_is_stored(client, admin_headers):
    res = _upload(client, admin_headers, "a.pdf", b"not a pdf", "application/pdf")
    assert res.status_code == 415
    assert "not a PDF" in res.json()["detail"]
    assert client.get("/api/documents", headers=admin_headers).json()["total"] == 0


def test_a_docx_renamed_to_pptx_is_415(client, admin_headers):
    res = _upload(client, admin_headers, "deck.pptx", make_docx(["hello"]))
    assert res.status_code == 415


def test_binary_named_txt_is_415(client, admin_headers):
    res = _upload(client, admin_headers, "notes.txt", b"abc\x00\x00def")
    assert res.status_code == 415


def test_unknown_extension_stays_400(client, admin_headers):
    assert _upload(client, admin_headers, "a.exe", b"x").status_code == 400


def test_over_the_limit_with_content_length_is_413_before_the_route_runs(
    client, admin_headers, limits
):
    limits.max_upload_mb = 1
    big = b"a" * (3 * 1024 * 1024)
    res = _upload(client, admin_headers, "big.txt", big)
    assert res.status_code == 413
    assert res.headers["x-request-id"]


def test_a_chunked_body_over_the_limit_is_413(client, admin_headers, limits):
    """No Content-Length (chunked transfer): the body is counted as it arrives."""
    limits.max_upload_mb = 1
    boundary = "bnd"
    head = (
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="big.txt"\r\n'
        "Content-Type: text/plain\r\n\r\n"
    ).encode()
    tail = f"\r\n--{boundary}--\r\n".encode()

    def body():
        yield head
        for _ in range(3):
            yield b"a" * (1024 * 1024)
        yield tail

    res = client.post(
        "/api/documents/upload",
        content=body(),
        headers={**admin_headers, "Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    assert res.status_code == 413


def test_the_size_limit_only_applies_to_uploads(client, admin_headers, limits):
    limits.max_upload_mb = 1
    res = client.post(
        "/api/search/semantic",
        json={"query": "x" * (2 * 1024 * 1024)},
        headers=admin_headers,
    )
    assert res.status_code != 413


def _run(middleware, scope, chunks):
    """Drive an ASGI middleware with a body in ``chunks``; return the sent messages."""
    import asyncio

    sent: list[dict] = []
    pending = list(chunks)

    async def receive():
        if pending:
            chunk = pending.pop(0)
            return {"type": "http.request", "body": chunk, "more_body": bool(pending)}
        return {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    asyncio.run(middleware(scope, receive, send))
    return sent


def _scope(path="/api/documents/upload", headers=()):
    return {"type": "http", "method": "POST", "path": path, "headers": list(headers)}


def test_middleware_refuses_a_declared_oversize_body_without_calling_the_app(limits):
    from ragfabric_server.middleware import UploadSizeLimitMiddleware

    limits.max_upload_mb = 1
    called = []

    async def app(scope, receive, send):
        called.append(True)

    sent = _run(
        UploadSizeLimitMiddleware(app),
        _scope(headers=[(b"content-length", str(5 * 1024 * 1024).encode())]),
        [b""],
    )
    assert called == []
    assert sent[0]["status"] == 413


def test_middleware_stops_a_chunked_body_at_the_limit(limits):
    from fastapi import HTTPException

    from ragfabric_server.middleware import UploadSizeLimitMiddleware

    limits.max_upload_mb = 1
    read: list[int] = []

    async def app(scope, receive, send):
        while True:
            message = await receive()
            read.append(len(message.get("body", b"")))
            if not message.get("more_body"):
                break

    with pytest.raises(HTTPException) as info:
        _run(UploadSizeLimitMiddleware(app), _scope(), [b"a" * (1024 * 1024)] * 5)
    assert info.value.status_code == 413
    # Stopped once past limit + slack (2 MB), never reading all five chunks.
    assert len(read) < 5


def test_middleware_leaves_other_paths_alone(limits):
    from ragfabric_server.middleware import UploadSizeLimitMiddleware

    limits.max_upload_mb = 1
    called = []

    async def app(scope, receive, send):
        called.append(True)

    _run(
        UploadSizeLimitMiddleware(app),
        _scope(path="/api/ask", headers=[(b"content-length", str(5 * 1024 * 1024).encode())]),
        [b""],
    )
    assert called == [True]
