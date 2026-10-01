"""ragfabric ask: ask a running RagFabric server a question.

Goes through the SDK over HTTP rather than calling core in process: the
normal case is a remote server, and a second in process code path would
drift from what API users experience. The cost is that this command needs a
running server to talk to; the help text says so.
"""

from __future__ import annotations

import os
from enum import StrEnum

import typer

from ragfabric_cli.ui.panels import render_answer
from ragfabric_sdk import Client
from ragfabric_sdk.errors import RagFabricError

DEFAULT_URL = "http://localhost:8000"


class Strategy(StrEnum):
    """Retrieval strategies this command can ask the server for.

    An Enum rather than a free string so Typer refuses an unknown name while
    parsing, before the command body runs. A typo then costs nothing: no
    request leaves the machine, no embedding call is spent, and the error
    names the valid choices instead of arriving as a 422 from the server.
    The members mirror the server's own Literal in schemas/search.py; the
    server stays the authority and still validates what it is sent.
    """

    auto = "auto"
    traditional = "traditional"
    vectorless = "vectorless"
    agentic = "agentic"
    graph = "graph"


def ask(
    question: str = typer.Argument(..., help="The question to ask."),
    url: str = typer.Option(
        None, "--url", help=f"Server URL (env RAGFABRIC_URL, default {DEFAULT_URL})."
    ),
    token: str = typer.Option(None, "--token", help="JWT bearer token (env RAGFABRIC_TOKEN)."),
    api_key: str = typer.Option(None, "--api-key", help="rf_ API key (env RAGFABRIC_API_KEY)."),
    top_k: int = typer.Option(8, "--top-k", min=1, max=50, help="Chunks to retrieve."),
    threshold: float = typer.Option(
        0.0, "--threshold", min=0.0, max=1.0, help="Similarity threshold."
    ),
    collection: int = typer.Option(None, "--collection", help="Collection id to search."),
    strategy: Strategy | None = typer.Option(
        None,
        "--strategy",
        help=(
            "Retrieval strategy. When omitted, the server's router.mode decides. "
            "auto lets the server's router choose one for the question; traditional "
            "embeds the question and searches the vector index; vectorless ranks with "
            "BM25 fused with ts_rank_cd and never calls an embedding model; agentic "
            "splits the question into parts, retrieves per part, and repairs or abandons "
            "the parts it cannot answer, reporting which those were; graph walks the knowledge "
            "graph from the entities the question names and prints the relationships "
            "it walked and any relationship claims the citation contract dropped."
        ),
    ),
    no_stream: bool = typer.Option(
        False, "--no-stream", help="Wait for the whole answer instead of streaming tokens."
    ),
    as_json: bool = typer.Option(
        False, "--json", help="Print the whole answer payload as JSON (implies --no-stream)."
    ),
) -> None:
    """Ask a question and print the cited answer.

    Needs a running RagFabric server: this command talks to it over HTTP
    through the RagFabric SDK, it does not answer the question locally.
    Streamed stdout can contain a stale, superseded draft ahead of the
    corrected answer (bytes already printed cannot be recalled); a machine
    consumer should use --json or --no-stream instead of parsing the stream.
    """
    url = url or os.environ.get("RAGFABRIC_URL") or DEFAULT_URL
    token = token or os.environ.get("RAGFABRIC_TOKEN")
    api_key = api_key or os.environ.get("RAGFABRIC_API_KEY")
    if not token and not api_key:
        typer.echo(
            "no credentials: pass --token or --api-key, or set RAGFABRIC_TOKEN "
            "or RAGFABRIC_API_KEY",
            err=True,
        )
        raise typer.Exit(2)

    params: dict[str, object] = {
        "top_k": top_k,
        "similarity_threshold": threshold,
    }
    if strategy is not None:
        params["strategy"] = strategy.value
    if collection is not None:
        params["collection_id"] = collection

    client = Client(url, token=token, api_key=api_key)
    try:
        if as_json or no_stream:
            answer = client.ask(question, **params)
            if as_json:
                typer.echo(answer.model_dump_json(indent=2))
            else:
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
            return

        citations: list[dict] = []
        subgraph: dict | None = None
        ran: str | None = None
        router: dict | None = None
        fallback_from: str | None = None
        dropped_relationship_claims: list[dict] = []
        run_id = None
        latency_ms = None
        for event in client.ask_stream(question, **params):
            if event.event == "retrieval":
                subgraph = event.data.get("subgraph")
                ran = event.data.get("strategy")
                router = event.data.get("router")
                fallback_from = event.data.get("fallback_from")
            elif event.event == "token":
                typer.echo(event.data.get("text", ""), nl=False)
            elif event.event == "superseded":
                # The caller already saw the rejected tokens printed above,
                # and there is no way to un-print a terminal: say so on
                # stderr (never mixed into a piped stdout file) and then put
                # the full corrected answer on stdout so a script reading
                # stdout still ends up with the complete, correct text, even
                # though it is preceded by the stale draft.
                typer.echo(
                    "\nnotice: the streamed answer above failed the citation "
                    "contract and was corrected; the corrected answer follows.",
                    err=True,
                )
                typer.echo("\n")
                typer.echo(event.data.get("text", ""), nl=False)
                dropped_relationship_claims = event.data.get("dropped_relationship_claims", [])
            elif event.event == "citations":
                citations = event.data.get("citations", [])
            elif event.event == "done":
                run_id = event.data.get("run_id")
                latency_ms = event.data.get("latency_ms")
        typer.echo("")
        render_answer(
            None,
            citations,
            strategy=ran,
            router=router,
            fallback_from=fallback_from,
            subgraph=subgraph,
            dropped_claims=[],
            dropped_relationship_claims=dropped_relationship_claims,
        )
        if run_id is not None:
            typer.echo(f"run {run_id} in {latency_ms}ms")
    except RagFabricError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc
    except Exception as exc:  # connection refused, DNS failure, timeout
        typer.echo(f"could not reach {url}: {exc}", err=True)
        raise typer.Exit(1) from exc
    finally:
        client.close()
