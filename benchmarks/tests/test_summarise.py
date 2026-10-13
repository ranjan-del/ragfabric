"""benchmarks/summarise.py turns stored runs into the tables the results pages publish."""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from summarise import markdown_table, run_row  # noqa: E402


def _run(**summary):
    base = {
        "status": "finished",
        "questions": 50,
        "errors": 1,
        "refusals": 3,
        "metrics": {
            "hit": 0.8,
            "reciprocal_rank": 0.61234,
            "recall": 0.7,
            "correctness": None,
            "citation_correct": 0.5,
        },
        "latency_ms": {"p50": 1200, "p95": 9000},
        "meta": {"commit": "abc1234", "judge_metrics": ["correctness"]},
    }
    base.update(summary)
    return SimpleNamespace(id=7, strategy="traditional", llm_model="llama3.2:3b", summary=base)


def test_run_row_reads_the_summary_and_keeps_unmeasured_as_none():
    row = run_row(_run())
    assert row["target"] == "traditional" and row["run"] == 7
    assert row["hit"] == 0.8 and row["mrr"] == 0.61234 and row["correctness"] is None
    assert row["refusals"] == 3 and row["p50_ms"] == 1200 and row["commit"] == "abc1234"


def test_a_skipped_run_says_why():
    row = run_row(_run(status="skipped", skipped="graph_store.enabled is false"))
    assert row["status"] == "skipped: graph_store.enabled is false"


def test_markdown_table_prints_n_a_for_none_and_two_decimals():
    text = markdown_table([run_row(_run())], ["target", "hit", "mrr", "correctness", "run"])
    lines = text.splitlines()
    assert lines[0] == "| target | hit | mrr | correctness | run |"
    assert lines[2] == "| traditional | 0.80 | 0.61 | n/a | 7 |"


# --- filling the results pages ---------------------------------------------------------------

from summarise import fill_markdown  # noqa: E402

PAGE = """# Results

<!-- summarise kind="batch" batch="b1" targets="traditional,vectorless" columns="target,hit,refusals,run" -->
stale text that the fill replaces
<!-- /summarise -->

Prose between tables is kept.

<!-- summarise kind="category" batch="b1" metric="correctness" targets="traditional,vectorless" -->
<!-- /summarise -->

<!-- summarise kind="leak" file="leak.json" targets="ragfabric:traditional,langchain:default" -->
<!-- /summarise -->
"""


def _row(target, run_id, **summary):
    run = _run(**summary)
    run.strategy, run.id = target, run_id
    return run_row(run)


def _filled(rows=None, leak=None):
    return fill_markdown(PAGE, rows=lambda batch: (rows or {}).get(batch, []), leak=lambda f: leak)


def test_with_nothing_stored_every_cell_is_tbd_and_the_page_keeps_its_markers():
    out = _filled()
    assert "stale text" not in out
    assert "| traditional | TBD | TBD | TBD |" in out
    assert "| vectorless | TBD | TBD | TBD |" in out
    assert "| multi_hop | TBD | TBD |" not in out  # categories unknown until a run is stored
    assert "| ragfabric:traditional | TBD | TBD | TBD |" in out
    assert out.count("<!-- summarise ") == 3 and out.count("<!-- /summarise -->") == 3
    assert "Prose between tables is kept." in out


def test_a_stored_run_fills_its_row_and_a_missing_target_stays_tbd():
    rows = {"b1": [_row("traditional", 13)]}
    out = _filled(rows)
    assert "| traditional | 0.80 | 3 | 13 |" in out
    assert "| vectorless | TBD | TBD | TBD |" in out


def test_a_stored_target_not_listed_is_appended_not_dropped():
    rows = {"b1": [_row("agentic", 16)]}
    out = _filled(rows)
    assert "| agentic | 0.80 | 3 | 16 |" in out


def test_unmeasured_is_n_a_not_tbd():
    rows = {"b1": [_row("traditional", 13, refusals=None)]}
    assert "| traditional | 0.80 | n/a | 13 |" in _filled(rows)


def test_a_run_still_in_progress_is_pending_not_unmeasured():
    rows = {"b1": [_row("agentic", 17, status="running", refusals=None, metrics={})]}
    assert "| agentic | TBD | TBD | 17 |" in _filled(rows)


def test_category_table_reads_by_category_per_target():
    by_cat = {"multi_hop": {"correctness": 0.6}, "ambiguous": {"correctness": None}}
    rows = {"b1": [_row("traditional", 13, by_category=by_cat)]}
    out = _filled(rows)
    assert "| category | traditional | vectorless |" in out
    assert "| ambiguous | n/a | TBD |" in out
    assert "| multi_hop | 0.60 | TBD |" in out


def test_leak_table_reads_the_leak_report():
    leak = {
        "targets": {
            "ragfabric:traditional": {
                "probes_leaking": 0,
                "forbidden_chunks": 0,
                "unfilled_slots": 2,
                "errors": [],
            }
        }
    }
    out = _filled(leak=leak)
    assert "| target | probes_leaking | forbidden_chunks | unfilled_slots |" in out
    assert "| ragfabric:traditional | 0 | 0 | 2 |" in out
    assert "| langchain:default | TBD | TBD | TBD |" in out


def test_filling_twice_gives_the_same_page():
    rows = {"b1": [_row("traditional", 13)]}
    once = _filled(rows)
    again = fill_markdown(once, rows=lambda b: rows.get(b, []), leak=lambda f: None)
    assert once == again
