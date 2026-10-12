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
