"""Turn stored evaluation batches into Markdown tables (every number names its run, ADR 0004).

    uv run python benchmarks/summarise.py <batch> [<batch> ...] [--columns target,hit,mrr,...]
    uv run python benchmarks/summarise.py --fill docs/benchmarks/comparison.md \
        docs/benchmarks/results/*.md [--leak-dir DIR]

Reads DATABASE_URL like the CLI (source benchmarks/env.sh). The first form prints one table per
batch. ``--fill`` rewrites, in place, every table a page marks with

    <!-- summarise kind="batch" batch="<batch>" targets="a,b" columns="target,hit,..." -->
    <!-- /summarise -->

(``kind="category"`` with ``metric="correctness"`` gives categories by target, and
``kind="leak"`` with ``file="leak-test.json"`` reads the access leak report from ``--leak-dir``).
A target or report with nothing stored is ``TBD``; a stored value that was not measured is
``n/a``. Nothing between the markers is written by hand, so no number can enter a page without a
stored run behind it.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections.abc import Callable
from pathlib import Path

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
        "judge": getattr(run, "judge_model", None),
        "commit": (s.get("meta") or {}).get("commit"),
        "date": _date(getattr(run, "started_at", None)),
        "run": run.id,
        "answered_by": (s.get("meta") or {}).get("answered_by"),
        "answer_source": (s.get("meta") or {}).get("answer_source"),
        "prompt_hash": ((s.get("meta") or {}).get("prompt_file_sha256") or "")[:12] or None,
        "export_date": (s.get("meta") or {}).get("export_date"),
        "by_category": s.get("by_category") or {},
    }


def _date(value) -> str | None:
    return value.date().isoformat() if value is not None else None


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


TBD = "TBD"
LEAK_COLUMNS = ["target", "probes_leaking", "forbidden_chunks", "unfilled_slots"]

_BLOCK = re.compile(
    r"(?P<open><!-- summarise (?P<attrs>[^>]*?)-->)\n(?P<body>.*?)(?P<close><!-- /summarise -->)",
    re.DOTALL,
)
_ATTR = re.compile(r'(\w+)="([^"]*)"')


def _split(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def _table(columns: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def _ordered(listed: list[str], stored: dict) -> list[str]:
    """Listed targets in their order, then any stored target the page did not list."""
    return listed + [t for t in stored if t not in listed]


def _batch_block(attrs: dict, rows: list[dict]) -> str:
    columns = _split(attrs.get("columns", ",".join(DEFAULT_COLUMNS)))
    by_target = {r["target"]: r for r in rows}
    out = []
    for target in _ordered(_split(attrs.get("targets", "")), by_target):
        row = by_target.get(target)
        out.append([target if c == "target" else _batch_cell(row, c) for c in columns])
    table = _table(columns, out)
    outside = [r for r in rows if r.get("answer_source")]
    if not outside:
        return table
    # Answers written outside RagFabric never appear without who wrote them: this table is added
    # to every batch block that holds such a run, whatever columns the page asked for.
    provenance = _table(
        ["target", "answered_by", "answer_source", "prompt_hash", "export_date"],
        [
            [_cell(r.get(c)) for c in ("target", "answered_by", "answer_source", "prompt_hash")]
            + [_cell(r.get("export_date"))]
            for r in outside
        ],
    )
    note = (
        "Answers below were written outside RagFabric and scored by it. They are not a "
        "controlled run and not a model tested through its API:"
    )
    return f"{table}\n\n{note}\n\n{provenance}"


def _batch_cell(row: dict | None, column: str) -> str:
    if row is None:
        return TBD
    value = row.get(column)
    # A run still in progress has no summary yet: its numbers are pending, not unmeasured.
    if value is None and row.get("status") == "running":
        return TBD
    return _cell(value)


def _category_block(attrs: dict, rows: list[dict]) -> str:
    metric = attrs.get("metric", "correctness")
    by_target = {r["target"]: r for r in rows}
    targets = _ordered(_split(attrs.get("targets", "")), by_target)
    categories = sorted({c for r in rows for c in (r.get("by_category") or {})})
    out = []
    for category in categories:
        cells = [category]
        for target in targets:
            row = by_target.get(target)
            cats = (row or {}).get("by_category") or {}
            cells.append(_cell((cats.get(category) or {}).get(metric)) if row else TBD)
        out.append(cells)
    if not out:  # categories are known only once a run is stored
        out.append([TBD] * (len(targets) + 1))
    return _table(["category", *targets], out)


def _leak_block(attrs: dict, report: dict | None) -> str:
    stored = (report or {}).get("targets") or {}
    out = []
    for target in _ordered(_split(attrs.get("targets", "")), stored):
        entry = stored.get(target)
        out.append([target] + [_cell(entry.get(c)) if entry else TBD for c in LEAK_COLUMNS[1:]])
    return _table(LEAK_COLUMNS, out)


def fill_markdown(
    text: str,
    rows: Callable[[str], list[dict]],
    leak: Callable[[str], dict | None],
) -> str:
    """Rewrite every marked table from stored runs; ``TBD`` where nothing is stored yet."""

    def replace(match: re.Match) -> str:
        attrs = dict(_ATTR.findall(match["attrs"]))
        kind = attrs.get("kind", "batch")
        if kind == "leak":
            body = _leak_block(attrs, leak(attrs.get("file", "leak-test.json")))
        elif kind == "category":
            body = _category_block(attrs, rows(attrs["batch"]))
        else:
            body = _batch_block(attrs, rows(attrs["batch"]))
        return f"{match['open']}\n{body}\n{match['close']}"

    return _BLOCK.sub(replace, text)


def _fill_files(paths: list[Path], leak_dir: Path) -> None:
    import json

    from ragfabric_core.evaluation.store import batch_runs
    from ragfabric_core.runtime import get_session_factory

    def load_leak(name: str) -> dict | None:
        path = leak_dir / name
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None

    with get_session_factory()() as db:

        def load_rows(batch: str) -> list[dict]:
            return [run_row(r) for r, _ in batch_runs(db, batch)]

        for path in paths:
            text = path.read_text(encoding="utf-8")
            path.write_text(fill_markdown(text, load_rows, load_leak), encoding="utf-8")
            print(f"filled {path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("batches", nargs="*")
    parser.add_argument("--columns", default=",".join(DEFAULT_COLUMNS))
    parser.add_argument("--fill", nargs="+", type=Path, metavar="PAGE")
    parser.add_argument(
        "--leak-dir",
        type=Path,
        default=Path(os.environ.get("RFB_BENCH_DIR", "~/ragfabric-bench")).expanduser(),
    )
    args = parser.parse_args(argv)
    if args.fill:
        _fill_files(args.fill, args.leak_dir)
        return 0
    if not args.batches:
        parser.error("name a batch, or --fill a page")
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
