"""The Markdown report of one stored batch."""

from test_eval_runner import GOOD, QS, Fake  # noqa: E402

from ragfabric_core.evaluation.judge import LexicalJudge
from ragfabric_core.evaluation.report import render_report
from ragfabric_core.evaluation.runner import run_batch
from ragfabric_core.evaluation.store import batch_router_agreement, batch_runs


def _render(factory, batch, agreement=True):
    with factory() as db:
        runs = batch_runs(db, batch)
        return render_report(runs, agreement=batch_router_agreement(runs) if agreement else None)


def test_the_header_names_the_run_and_says_it_is_one_run(eval_db):
    _, factory = eval_db
    run_batch(
        [Fake("traditional", GOOD)],
        QS,
        LexicalJudge(),
        factory,
        batch="r1",
        meta={"commit": "abc123", "top_k": 5, "llm_provider": "offline"},
        llm_model="scripted",
        embedding_model="hashing",
    )
    text = _render(factory, "r1")
    assert "one run on one setup, not a benchmark" in text
    assert "`abc123`" in text and "offline / scripted" in text and "hashing" in text
    assert "lexical word overlap" in text
    assert "| traditional | 3 | 0 | 1 |" in text  # q3 is a refusal
    assert "Est. cost" in text and "estimate from configured pricing" in text
    assert "## Router" not in text
    assert "| ambiguous | n/a |" in text  # no expected sources: unmeasured hit rate
    assert chr(0x2014) not in text  # no em dash


def test_skipped_targets_and_the_router_section(eval_db):
    _, factory = eval_db
    run_batch(
        [Fake("auto", GOOD), Fake("vectorless", GOOD)],
        QS,
        LexicalJudge(),
        factory,
        batch="r2",
        meta={},
        skipped=[("graph", "graph_store.enabled is false")],
    )
    text = _render(factory, "r2")
    assert "## Skipped targets" in text and "`graph`: graph_store.enabled is false" in text
    assert "## Router" in text and "matched the best fixed strategy" in text
    assert "| q1 | auto |" in text
