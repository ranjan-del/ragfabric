"""The leak test's counting, on plain values (D17)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from leak_test import ProbeResult, score_probe, summarise  # noqa: E402


def test_forbidden_chunks_are_counted_and_unfilled_slots_need_allowed_chunks_left():
    r = score_probe(retrieved=[1, 2, 2, 1], forbidden={2}, top_k=5, allowed_available=10)
    assert r == ProbeResult(retrieved=4, forbidden=2, unfilled=1)


def test_unfilled_slots_are_not_counted_when_nothing_allowed_was_left():
    r = score_probe(retrieved=[1, 1], forbidden={2}, top_k=5, allowed_available=2)
    assert r.unfilled == 0


def test_summary_counts_probes_with_any_leak():
    s = summarise([ProbeResult(5, 0, 0), ProbeResult(5, 2, 0), ProbeResult(3, 0, 2)])
    assert s == {
        "probes": 3,
        "probes_leaking": 1,
        "forbidden_chunks": 2,
        "unfilled_slots": 2,
        "retrieved_chunks": 13,
    }
