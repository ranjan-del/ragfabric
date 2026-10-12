"""Turn stored evaluation batches into Markdown tables (every number names its run, ADR 0004).

    uv run python benchmarks/summarise.py <batch> [<batch> ...] [--columns target,hit,mrr,...]

Reads DATABASE_URL like the CLI (source benchmarks/env.sh). Prints one table per batch.
"""

from __future__ import annotations

import argparse
import sys

DEFAULT_COLUMNS = [
    "target",
    "questions",
    "errors",
    "hit",
    "mrr",
    "recall",
    "correctness",
    "citation",
    "refusals",
    "p50_ms",
    "p95_ms",
    "run",
]


def run_row(run) -> dict:
    s = run.summary or {}
    m = s.get("metrics") or {}
    lat = s.get("latency_ms") or {}
    status = s.get("status")
    if status == "skipped":
        status = f"skipped: {s.get('skipped')}"
    return {
        "target": run.strategy,
        "status": status,
        "questions": s.get("questions"),
        "errors": s.get("errors"),
        "hit": m.get("hit"),
        "mrr": m.get("reciprocal_rank"),
        "recall": m.get("recall"),
        "precision": m.get("precision"),
        "correctness": m.get("correctness"),
        "citation": m.get("citation_correct"),
        "refusals": s.get("refusals"),
        "fallbacks": s.get("fallbacks"),
        "p50_ms": lat.get("p50"),
        "p95_ms": lat.get("p95"),
        "llm_model": run.llm_model,
        "commit": (s.get("meta") or {}).get("commit"),
        "run": run.id,
    }


def _cell(value) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def markdown_table(rows: list[dict], columns: list[str]) -> str:
    out = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    out += ["| " + " | ".join(_cell(r.get(c)) for c in columns) + " |" for r in rows]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("batches", nargs="+")
    parser.add_argument("--columns", default=",".join(DEFAULT_COLUMNS))
    args = parser.parse_args(argv)
    from ragfabric_core.evaluation.store import batch_runs
    from ragfabric_core.runtime import get_session_factory

    columns = args.columns.split(",")
    with get_session_factory()() as db:
        for batch in args.batches:
            runs = batch_runs(db, batch)
            if not runs:
                print(f"no batch {batch!r}", file=sys.stderr)
                continue
            print(f"### {batch}\n")
            print(markdown_table([run_row(r) for r, _ in runs], columns))
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
