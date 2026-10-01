"""Renderers. Rich on a terminal, the exact plain lines everywhere else."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import typer
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ragfabric_cli.ui import console


@dataclass(frozen=True)
class CheckView:
    name: str
    status: Literal["pass", "warn", "fail", "skip"]
    detail: str
    fix: str | None = None


@dataclass(frozen=True)
class StrategyRow:
    name: str
    best_at: str
    model_calls: str
    example: str


def _print_sources(citations: list[dict]) -> None:
    used = [citation for citation in citations if citation.get("used")]
    if not used:
        return
    typer.echo("sources:")
    for citation in used:
        name = citation.get("filename") or f"document {citation.get('document_id')}"
        page = f" p{citation['page']}" if citation.get("page") else ""
        typer.echo(f"  {citation['marker']} {name}{page}")


def _print_graph(subgraph: dict | None, dropped: list[dict]) -> None:
    """The relationships the graph strategy walked, and the claims about them it dropped.

    Printed as ``name RELATION name`` in the direction walked, the same reading
    the answer's ``[E k]`` markers were numbered against. Nothing is printed
    for a strategy that walks no graph.
    """
    if subgraph is not None:
        names = {node["id"]: node["name"] for node in subgraph.get("nodes", [])}
        edges = subgraph.get("edges", [])
        if edges:
            typer.echo("graph:")
            for number, edge in enumerate(edges, start=1):
                start, end = edge["source_id"], edge["target_id"]
                if edge.get("reversed"):
                    start, end = end, start
                typer.echo(
                    f"  [E {number}] {names.get(start, start)} {edge['walked_as']} "
                    f"{names.get(end, end)}"
                )
        elif subgraph.get("empty_reason"):
            typer.echo(f"graph: nothing walked ({subgraph['empty_reason']})")
        if subgraph.get("truncated"):
            typer.echo("graph: the walk was cut by the node budget")
    if dropped:
        typer.echo("dropped relationship claims:")
        for claim in dropped:
            typer.echo(f"  {claim['reason']}: {claim['text']}")


def _routing_line(strategy: str | None, router: dict | None, fallback_from: str | None) -> str:
    if router is None:
        return ""
    line = f"Strategy: {strategy or router.get('selected_strategy')} ({router.get('source')})."
    if router.get("reasoning"):
        line += f" {router['reasoning']}"
    if fallback_from:
        # Neutral on purpose: the routed strategy may have found nothing or may
        # have failed, and the response does not say which.
        line += f" Fell back from {fallback_from}."
    return line


def _print_routing(strategy: str | None, router: dict | None, fallback_from: str | None) -> None:
    """One line saying which strategy ran, why, and whether it fell back.

    Printed only when the server sent a router decision.
    """
    line = _routing_line(strategy, router, fallback_from)
    if line:
        typer.echo(line)


def render_answer(
    text: str | None,
    citations: list[dict],
    *,
    strategy: str | None,
    router: dict | None,
    fallback_from: str | None,
    subgraph: dict | None,
    dropped_claims: list[dict],
    dropped_relationship_claims: list[dict],
) -> None:
    """Print an answer. ``text`` is None when the tokens were already streamed.

    ``dropped_claims`` is reserved for the final rich layout; only the relationship
    claims are printed today, exactly as before.
    """
    if not console.is_rich():
        if text is not None:
            typer.echo(text)
            typer.echo("")
        _print_sources(citations)
        _print_graph(subgraph, dropped_relationship_claims)
        _print_routing(strategy, router, fallback_from)
        return

    out = console.get_console()
    if text is not None:
        out.print(Panel(Text(text), title="Answer", title_align="left"))
    used = [c for c in citations if c.get("used")]
    if used:
        sources = Table.grid(padding=(0, 1))
        for citation in used:
            name = citation.get("filename") or f"document {citation.get('document_id')}"
            page = f" p{citation['page']}" if citation.get("page") else ""
            sources.add_row(Text(str(citation["marker"])), Text(f"{name}{page}"))
        out.print(Text("sources", style="bold"))
        out.print(sources)
    line = _routing_line(strategy, router, fallback_from)
    if line:
        out.print(Text(line, style="dim"))


_STATUS_STYLE = {"pass": "green", "warn": "yellow", "fail": "red", "skip": "dim"}


def render_checks(results: list[CheckView]) -> None:
    if not console.is_rich():
        for r in results:
            typer.echo(f"{r.status:<5} {r.name}: {r.detail}")
            if r.fix and r.status in ("warn", "fail"):
                typer.echo(f"      fix: {r.fix}")
        return
    table = Table(show_header=False, box=None, padding=(0, 1))
    for r in results:
        detail = Text(r.detail)
        if r.fix and r.status in ("warn", "fail"):
            detail.append(f"\nfix: {r.fix}", style="dim")
        table.add_row(Text(r.status, style=_STATUS_STYLE[r.status]), Text(r.name), detail)
    console.get_console().print(table)


def render_strategies(rows: list[StrategyRow]) -> None:
    if not console.is_rich():
        for row in rows:
            typer.echo(f"{row.name}: {row.best_at} ({row.model_calls} model calls)")
            typer.echo(f"  e.g. {row.example}")
        return
    table = Table(title="Strategies")
    for column in ("Strategy", "Best at", "Model calls", "Example"):
        table.add_column(column, overflow="fold")
    for row in rows:
        table.add_row(Text(row.name), Text(row.best_at), Text(row.model_calls), Text(row.example))
    console.get_console().print(table)


def render_next_steps(steps: list[tuple[str, str]]) -> None:
    if not console.is_rich():
        typer.echo("next steps:")
        for command, why in steps:
            typer.echo(f"  {command}  # {why}")
        return
    table = Table.grid(padding=(0, 2))
    for command, why in steps:
        table.add_row(Text(command, style="bold"), Text(why, style="dim"))
    console.get_console().print(Panel(table, title="Next steps", title_align="left"))
