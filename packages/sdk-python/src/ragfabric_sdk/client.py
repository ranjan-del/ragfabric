"""HTTP client for a RagFabric server.

Talks HTTP only and never imports ragfabric_core, ragfabric_server or
ragfabric_cli: an adopter installs this on a laptop or in a serverless
function to call a RagFabric server someone else runs. If it imported core
it would drag SQLAlchemy, pgvector, chromadb and the migrations into that
environment, and a version skew between the client's core and the server's
core would produce failures that look like API bugs. The rule is enforced
by an import linter contract, not by convention.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import httpx

from ragfabric_sdk.errors import raise_for_status
from ragfabric_sdk.models import Answer, AskEvent, Document, Run, SearchResult

DEFAULT_TIMEOUT = 30.0


class Client:
    """A RagFabric API client.

    Args:
        base_url: e.g. "http://localhost:8000".
        token: a JWT from POST /api/auth/login; sent as Authorization: Bearer.
        api_key: an rf_ key; sent as X-API-Key. If both are given, the API
            key wins and the bearer header is not sent.
        timeout: seconds.
        transport: a testing seam only, not a public feature. Pass an
            ``httpx`` transport (for example ``httpx.MockTransport``) to
            exercise this client against a fake server, so tests never touch
            the network.
    """

    def __init__(
        self,
        base_url: str,
        token: str | None = None,
        api_key: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        headers = {}
        if api_key:
            headers["X-API-Key"] = api_key
        elif token:
            headers["Authorization"] = f"Bearer {token}"
        self._http = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers=headers,
            timeout=timeout,
            transport=transport,
        )

    def __enter__(self) -> Client:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._http.close()

    def ask(self, query: str, strategy: str = "traditional", **params) -> Answer:
        """POST /api/ask with stream=False and return the finished, cited answer.

        ``strategy`` names the retrieval strategy: ``"traditional"`` (embed the
        question, search the vector index), ``"vectorless"`` (BM25 fused with
        ts_rank_cd, no embedding call at all), ``"agentic"`` (decompose the
        question, retrieve per part, repair or abandon the parts that fail) or
        ``"graph"`` (see below).
        It is spelled out as a named parameter rather than left to ``**params``
        because it changes what the server does, and a caller should be able to
        find it in the signature. The server rejects any other name with a 422.

        With ``"agentic"``, the answer also carries ``sub_questions`` (what was
        and was not answered, and why), ``dropped_claims`` (claims the citation
        contract refused, removed rather than retried), ``dated_sources`` and
        the agent's ``trace``. Every one of them is empty for the other
        strategies, which have no parts to report.

        With ``"graph"`` (walk the knowledge graph from the entities the
        question names), the answer carries ``subgraph`` (the nodes and edges
        walked, whether the walk was truncated, and why it was empty if it
        was) and ``dropped_relationship_claims`` (relationship claims the graph
        citation contract removed, each with its reason). ``subgraph`` is None
        for every other strategy. A graph request with ``document_id`` or
        ``format`` is refused with a 422, since the walk cannot apply them.
        """
        res = self._http.post(
            "/api/ask",
            json={"query": query, "stream": False, "strategy": strategy, **params},
        )
        raise_for_status(res)
        return Answer.model_validate(res.json())

    def ask_stream(self, query: str, strategy: str = "traditional", **params) -> Iterator[AskEvent]:
        """POST /api/ask with stream=True and yield each server-sent event.

        Events arrive in this order: ``retrieval``, one or more ``token``,
        optionally ``superseded``, ``citations``, then ``done`` (carrying
        ``run_id`` and ``latency_ms``).

        ``superseded`` is not a failure: it fires when the streamed text
        already rendered to the caller turns out to violate the citation
        contract. The server cannot retake back the tokens it already sent,
        so instead of silently recording a different answer than what was
        shown, it repairs the answer and announces the repair. A caller
        handling this event should REPLACE whatever it has drawn from the
        preceding ``token`` events with ``event.data["text"]``, the same way
        the ``ragfabric ask`` command does.

        Per the SSE wire format, a ``data:`` line with no preceding
        ``event:`` line for that block defaults to the event name
        ``"message"``. The RagFabric server always sends an explicit
        ``event:`` line today, but a client library follows the spec rather
        than only the one server it was written against, so a bare
        ``data:`` line is still yielded rather than silently dropped.
        """
        with self._http.stream(
            "POST",
            "/api/ask",
            json={"query": query, "stream": True, "strategy": strategy, **params},
        ) as res:
            if res.status_code >= 400:
                res.read()
                raise_for_status(res)
            name: str | None = None
            for line in res.iter_lines():
                if line.startswith("event:"):
                    name = line.split(":", 1)[1].strip()
                elif line.startswith("data:"):
                    yield AskEvent(
                        event=name if name is not None else "message",
                        data=json.loads(line.split(":", 1)[1].strip()),
                    )
                    name = None

    def search(
        self,
        query: str,
        mode: str = "semantic",
        strategy: str = "traditional",
        **params,
    ) -> list[SearchResult]:
        """POST /api/search/semantic or /api/search/hybrid and return the ranked chunks.

        ``strategy`` selects the retrieval strategy, as on ``ask``. Note that
        ``mode="hybrid"`` fuses one vector ranking with one lexical ranking, so
        the server refuses any strategy that does not provide that pair with a
        422; use the default ``mode="semantic"`` to search with
        ``strategy="vectorless"``, ``strategy="agentic"`` or ``strategy="graph"``.
        """
        path = "/api/search/hybrid" if mode == "hybrid" else "/api/search/semantic"
        res = self._http.post(
            path, json={"query": query, "mode": mode, "strategy": strategy, **params}
        )
        raise_for_status(res)
        return [SearchResult.model_validate(r) for r in res.json()["results"]]

    def ingest(
        self,
        path: str | Path,
        collection: int | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> Document:
        """POST /api/documents/upload and return the ingested document.

        ``collection`` maps to the upload form's ``collection_id`` field.
        ``chunk_size``/``chunk_overlap`` are the optional per-upload chunking
        overrides the route accepts; left unset, the document is chunked
        with the server's configured defaults.
        """
        file_path = Path(path)
        data: dict[str, str] = {}
        if collection is not None:
            data["collection_id"] = str(collection)
        if chunk_size is not None:
            data["chunk_size"] = str(chunk_size)
        if chunk_overlap is not None:
            data["chunk_overlap"] = str(chunk_overlap)
        with file_path.open("rb") as handle:
            res = self._http.post(
                "/api/documents/upload",
                files={"file": (file_path.name, handle)},
                data=data or None,
            )
        raise_for_status(res)
        return Document.model_validate(res.json())

    def documents(self) -> list[Document]:
        """GET /api/documents. The route always returns the {"items": [...],
        "total": N} envelope (schemas.document.DocumentList), never a bare
        list, so that is the only shape parsed here."""
        res = self._http.get("/api/documents")
        raise_for_status(res)
        return [Document.model_validate(r) for r in res.json()["items"]]

    def run(self, run_id: int) -> Run:
        """GET /api/runs/{id}."""
        res = self._http.get(f"/api/runs/{run_id}")
        raise_for_status(res)
        return Run.model_validate(res.json())
