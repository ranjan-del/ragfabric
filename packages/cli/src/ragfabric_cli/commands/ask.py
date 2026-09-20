"""ragfabric ask: ask a running RagFabric server a question.

Goes through the SDK over HTTP rather than calling core in process: the
normal case is a remote server, and a second in process code path would
drift from what API users experience. The cost is that this command needs a
running server to talk to; the help text says so.
"""

from __future__ import annotations

import os

import typer

from ragfabric_sdk import Client
from ragfabric_sdk.errors import RagFabricError

DEFAULT_URL = "http://localhost:8000"


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

    params: dict[str, object] = {"top_k": top_k, "similarity_threshold": threshold}
    if collection is not None:
        params["collection_id"] = collection

    client = Client(url, token=token, api_key=api_key)
    try:
        if as_json or no_stream:
            answer = client.ask(question, **params)
            if as_json:
                typer.echo(answer.model_dump_json(indent=2))
            else:
                typer.echo(answer.answer)
                typer.echo("")
                _print_sources([citation.model_dump() for citation in answer.citations])
            return

        citations: list[dict] = []
        run_id = None
        latency_ms = None
        for event in client.ask_stream(question, **params):
            if event.event == "token":
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
            elif event.event == "citations":
                citations = event.data.get("citations", [])
            elif event.event == "done":
                run_id = event.data.get("run_id")
                latency_ms = event.data.get("latency_ms")
        typer.echo("")
        _print_sources(citations)
        if run_id is not None:
            typer.echo(f"run {run_id} in {latency_ms}ms")
    except RagFabricError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc
    except Exception as exc:  # connection refused, DNS failure, timeout
        typer.echo(f"could not reach {url}: {exc}", err=True)
        raise typer.Exit(1) from exc
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()


def _print_sources(citations: list[dict]) -> None:
    used = [citation for citation in citations if citation.get("used")]
    if not used:
        return
    typer.echo("sources:")
    for citation in used:
        name = citation.get("filename") or f"document {citation.get('document_id')}"
        page = f" p{citation['page']}" if citation.get("page") else ""
        typer.echo(f"  {citation['marker']} {name}{page}")
