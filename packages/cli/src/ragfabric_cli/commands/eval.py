"""``ragfabric eval``: load the corpus, run a batch, read stored runs, render the report.

Thin by design: the harness is ``ragfabric_core.evaluation``; this module
parses options and prints. Runs happen in process against the configured
database and providers, with no server (design decisions D1 and D16).
"""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import typer

from ragfabric_core import __version__
from ragfabric_core.evaluation.corpus import ingest_corpus
from ragfabric_core.evaluation.dataset import (
    QuestionSet,
    QuestionType,
    content_hash,
    load_questions,
    shipped_corpus_paths,
    shipped_questions_path,
)
from ragfabric_core.evaluation.report import render_report
from ragfabric_core.evaluation.runner import run_batch
from ragfabric_core.evaluation.session import prepare
from ragfabric_core.evaluation.store import (
    batch_router_agreement,
    batch_runs,
    get_run,
    latest_batch,
    list_runs,
    result_dict,
    run_dict,
)
from ragfabric_core.evaluation.strategy_target import DEFAULT_TARGETS
from ragfabric_core.runtime import get_config, get_session_factory

eval_app = typer.Typer(help="Measure the strategies on a question set.", no_args_is_help=True)


def _fail(message: str) -> None:
    typer.echo(message, err=True)
    raise typer.Exit(code=1)


def _commit() -> str:
    """The checkout's short commit, or empty when not run from a git checkout."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


def _fmt(value) -> str:
    return "n/a" if value is None else f"{value:.2f}"


@eval_app.command("corpus")
def corpus(
    collection: str | None = typer.Option(
        None, "--collection", help="Defaults to evaluation.collection."
    ),
) -> None:
    """Ingest the shipped synthetic corpus into the evaluation collection (idempotent).

    \b
    Examples:
      ragfabric eval corpus
    """
    name = collection or get_config().evaluation.collection
    ingested, skipped, failed = ingest_corpus(get_session_factory(), name, shipped_corpus_paths())
    typer.echo(
        f"{name}: {len(ingested)} ingested, {len(skipped)} already loaded, {len(failed)} failed"
    )
    if failed:
        _fail(f"failed: {', '.join(failed)}")


@eval_app.command("run")
def run(
    strategy: list[str] = typer.Option(
        None,
        "--strategy",
        help=f"Repeatable. Default: {', '.join(DEFAULT_TARGETS)}. "
        "Also traditional+rerank=<none|llm|cross_encoder>.",
    ),
    questions: Path | None = typer.Option(None, "--questions", help="Your own question file."),
    collection: str | None = typer.Option(None, "--collection"),
    category: list[str] = typer.Option(None, "--category", help="Repeatable question_type."),
    judge: str | None = typer.Option(None, "--judge", help="auto, llm or lexical."),
    report: Path | None = typer.Option(None, "--report", help="Write the Markdown report here."),
    batch: str | None = typer.Option(None, "--batch", help="Batch label; default eval-<time>."),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Run a question set through the strategies, score and store every answer.

    \b
    Examples:
      ragfabric eval run --report docs/benchmarks/latest.md
      ragfabric eval run --strategy graph --category multi_hop
      ragfabric eval run --questions ./my-questions.json --collection handbook
    """
    if judge is not None and judge not in ("auto", "llm", "lexical"):
        _fail(f"unknown judge {judge!r}: expected auto, llm or lexical")
    path = questions or shipped_questions_path()
    try:
        qs = load_questions(path)
        if category:
            unknown = [c for c in category if c not in set(QuestionType)]
            if unknown:
                raise ValueError(f"unknown category {', '.join(unknown)}")
            kept = [q for q in qs.questions if q.question_type in set(category)]
            if not kept:
                raise ValueError("no question matches the categories given")
            qs = QuestionSet(name=qs.name, description=qs.description, questions=kept)
        cfg = get_config()
        factory = get_session_factory()
        prepared = prepare(
            cfg,
            factory,
            specs=list(strategy or DEFAULT_TARGETS),
            collection=collection,
            judge=judge,
        )
    except ValueError as exc:
        _fail(str(exc))
    label = batch or f"eval-{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
    shipped_corpus = questions is None and prepared.collection == cfg.evaluation.collection
    meta = {
        "commit": _commit(),
        "ragfabric_version": __version__,
        "questions_hash": content_hash([path]),
        "corpus_hash": content_hash(shipped_corpus_paths()) if shipped_corpus else "own corpus",
        "collection": prepared.collection,
        "categories": sorted(category) if category else None,
        **prepared.meta,
    }
    if not as_json:
        typer.echo(
            f"batch {label}: {len(qs.questions)} questions x {len(prepared.targets)} targets, "
            f"judge {prepared.judge.kind}"
            + (f" ({prepared.judge.model})" if prepared.judge.model else "")
        )
        for name, reason in prepared.skipped:
            typer.echo(f"skipped {name}: {reason}")

    def progress(target: str, row) -> None:
        if as_json:
            return
        error = row.details.get("error")
        typer.echo(
            f"  {target} {row.question_id}: "
            + (
                f"error {error}"
                if error
                else f"hit {_fmt(row.hit)} correctness {_fmt(row.correctness)} "
                f"citation {_fmt(row.citation_correct)} {row.latency_ms}ms"
            )
        )

    run_ids = run_batch(
        prepared.targets,
        qs,
        prepared.judge,
        factory,
        batch=label,
        meta=meta,
        skipped=prepared.skipped,
        llm_model=prepared.llm_model,
        embedding_model=prepared.embedding_model,
        on_result=progress,
    )
    with factory() as db:
        runs = batch_runs(db, label)
        if report is not None:
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text(
                render_report(runs, agreement=batch_router_agreement(runs)), encoding="utf-8"
            )
        if as_json:
            typer.echo(json.dumps({"batch": label, "runs": [run_dict(r) for r, _ in runs]}))
            return
        typer.echo("")
        for run_row, _ in runs:
            s = run_row.summary
            if s.get("status") == "skipped":
                continue
            m = s["metrics"]
            typer.echo(
                f"{run_row.strategy:<32} hit {_fmt(m['hit'])}  mrr {_fmt(m['reciprocal_rank'])}  "
                f"correctness {_fmt(m['correctness'])}  citation {_fmt(m['citation_correct'])}  "
                f"p50 {s['latency_ms']['p50']}ms  errors {s['errors']}"
            )
    typer.echo(f"stored as runs {', '.join(str(i) for i in run_ids)} (batch {label})")
    if report is not None:
        typer.echo(f"report written to {report}")


@eval_app.command("list")
def list_command(
    limit: int = typer.Option(20, "--limit", min=1),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """List recent evaluation runs.

    \b
    Examples:
      ragfabric eval list
      ragfabric eval list --limit 5 --json
    """
    with get_session_factory()() as db:
        runs = [run_dict(r) for r in list_runs(db, limit)]
    if as_json:
        typer.echo(json.dumps(runs))
        return
    if not runs:
        typer.echo("no evaluation runs yet: ragfabric eval run")
        return
    for r in runs:
        s = r["summary"]
        state = s.get("status")
        hit = (s.get("metrics") or {}).get("hit")
        typer.echo(
            f"{r['id']:>5}  {r['batch']}  {r['target']:<28} {state:<9} "
            f"questions {s.get('questions', 0)}  hit {_fmt(hit)}"
        )


@eval_app.command("show")
def show(run_id: int, as_json: bool = typer.Option(False, "--json")) -> None:
    """Show one run's summary and its per question rows.

    \b
    Examples:
      ragfabric eval show 1
      ragfabric eval show 1 --json
    """
    with get_session_factory()() as db:
        found = get_run(db, run_id)
        if found is None:
            _fail(f"no evaluation run {run_id}")
        run_row, rows = found
        data = {**run_dict(run_row), "results": [result_dict(r) for r in rows]}
    if as_json:
        typer.echo(json.dumps(data))
        return
    s = data["summary"]
    typer.echo(f"run {data['id']} {data['target']} (batch {data['batch']}): {s.get('status')}")
    if s.get("skipped"):
        typer.echo(f"skipped: {s['skipped']}")
    for r in data["results"]:
        error = r["details"].get("error")
        typer.echo(
            f"  {r['question_id']} [{r['question_type']}] "
            + (
                f"error {error}"
                if error
                else f"hit {_fmt(r['hit'])} correctness {_fmt(r['correctness'])} "
                f"citation {_fmt(r['citation_correct'])}"
            )
        )


@eval_app.command("report")
def report_command(
    batch: str | None = typer.Argument(None, help="Batch label; default the latest."),
    out: Path | None = typer.Option(None, "--out", help="Write here instead of printing."),
) -> None:
    """Render a stored batch as Markdown.

    \b
    Examples:
      ragfabric eval report
      ragfabric eval report --out docs/benchmarks/latest.md
    """
    with get_session_factory()() as db:
        label = batch or latest_batch(db)
        runs = batch_runs(db, label) if label else []
        if not runs:
            _fail(f"no evaluation batch {label!r}" if label else "no evaluation runs yet")
        text = render_report(runs, agreement=batch_router_agreement(runs))
    if out is None:
        typer.echo(text)
    else:
        out.write_text(text, encoding="utf-8")
        typer.echo(f"report written to {out}")
