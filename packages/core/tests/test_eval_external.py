"""Scoring answers written outside RagFabric: export, import, errors and provenance."""

import json
from pathlib import Path

import pytest

from ragfabric_core.config_file import load_config
from ragfabric_core.evaluation.corpus import collection_id, ingest_corpus
from ragfabric_core.evaluation.dataset import QuestionSet
from ragfabric_core.evaluation.external import (
    PromptFileError,
    export_header,
    export_prompts,
    import_answers,
    read_answers,
    read_prompts,
    slug,
    target_name,
)
from ragfabric_core.evaluation.judge import LexicalJudge
from ragfabric_core.evaluation.report import render_report
from ragfabric_core.evaluation.store import batch_runs
from ragfabric_core.evaluation.strategy_target import build_targets
from ragfabric_core.generate.cited import NO_EVIDENCE_ANSWER, SYSTEM_PROMPT, build_prompt
from ragfabric_core.providers.base import Completion
from ragfabric_core.strategies.base import RetrievedChunk
from ragfabric_core.strategies.registry_defaults import default_registry

QS = QuestionSet.model_validate(
    {
        "name": "tiny",
        "questions": [
            {
                "id": "q1",
                "question": "How many days of casual leave?",
                "expected_answer": "10 days of casual leave",
                "expected_sources": [{"document": "leave.md", "evidence": "10 days"}],
                "question_type": "simple_factual",
            },
            {
                "id": "q2",
                "question": "What is the hotel limit per night?",
                "expected_answer": "180 US dollars per night",
                "expected_sources": [{"document": "travel.md"}],
                "question_type": "exact_match",
            },
            {
                "id": "q3",
                "question": "Who founded the company?",
                "expected_answer": "",
                "question_type": "ambiguous",
            },
        ],
    }
)


class Spy:
    """A scripted model that records the messages it was sent, as a local model would see them."""

    name = "scripted"
    default_model = "scripted-1"

    def __init__(self, text="Employees receive 10 days of casual leave [1]."):
        self.text = text
        self.sent = []

    def complete(self, messages, *, model=None, max_tokens=1024, temperature=0.0, json_schema=None):
        self.sent.append(messages)
        return Completion(
            text=self.text,
            model="scripted-1",
            provider="scripted",
            input_tokens=7,
            output_tokens=3,
            latency_ms=1,
        )


@pytest.fixture()
def loaded(eval_db, tmp_path):
    cfg_path, factory = eval_db
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "leave.md").write_text("# Leave\n\nEmployees receive 10 days of casual leave per year.")
    (docs / "travel.md").write_text("# Travel\n\nThe hotel limit is 180 US dollars per night.")
    ingest_corpus(factory, "eval", [docs / "leave.md", docs / "travel.md"])
    cfg = load_config(cfg_path)
    with factory() as db:
        cid = collection_id(db, "eval")
    return cfg, factory, cid


def _target(loaded, spec, llm):
    cfg, factory, cid = loaded
    targets, _ = build_targets(
        [spec],
        cfg,
        default_registry(cfg, factory),
        llm,
        collection_id=cid,
        session_factory=factory,
    )
    return targets[0]


def _export(loaded, spec, tmp_path, llm=None):
    target = _target(loaded, spec, llm or Spy())
    out = tmp_path / f"{spec}.jsonl"
    header = export_header(
        corpus="own corpus", collection="eval", top_k=5, strategy=spec, commit="abc1234"
    )
    export_prompts(QS, target, out, header=header)
    return out


def _lines(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.mark.parametrize("spec", ["traditional", "vectorless"])
def test_export_messages_equal_what_a_normal_run_sends(loaded, tmp_path, spec):
    spy = Spy()
    target = _target(loaded, spec, spy)
    out = _export(loaded, spec, tmp_path, spy)
    assert spy.sent == [], "export must not call a model"
    header, *records = _lines(out)
    assert header["type"] == "header" and header["strategy"] == spec
    assert header["collection"] == "eval" and header["top_k"] == 5 and header["commit"] == "abc1234"
    assert header["exported_at"]
    for record in records:
        target.answer(record["question"])
    assert len(spy.sent) == len(records) == 3
    for record, sent in zip(records, spy.sent, strict=True):
        assert record["messages"] == [m.model_dump() for m in sent]
        chunks = [
            RetrievedChunk(
                chunk_id=p["chunk_id"], document_id=0, collection_id=None, text=p["text"]
            )
            for p in record["passages"]
        ]
        assert record["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT}
        assert record["messages"][1]["content"] == build_prompt(record["question"], chunks)
        assert [p["number"] for p in record["passages"]] == list(range(1, len(chunks) + 1))
        assert all(p["document"] and isinstance(p["chunk_id"], int) for p in record["passages"])


def test_export_carries_the_retrieval_metric_inputs(loaded, tmp_path):
    out = _export(loaded, "traditional", tmp_path)
    _, q1, _, q3 = _lines(out)
    assert q1["ground_truth"]["expected_sources"] == [
        {"document": "leave.md", "evidence": "10 days"}
    ]
    assert q1["ground_truth"]["question_type"] == "simple_factual"
    assert q3["ground_truth"]["expected_sources"] == []


def test_export_refuses_other_strategies(loaded, tmp_path):
    from ragfabric_core.evaluation.external import ExportError

    target = _target(loaded, "traditional", Spy())
    target.name = "graph"
    with pytest.raises(ExportError):
        export_prompts(QS, target, tmp_path / "x.jsonl", header={})


def _write_answers(path: Path, lines):
    path.write_text("\n".join(lines) + "\n")
    return path


GOOD = [
    json.dumps({"id": "q1", "answer": "Employees receive 10 days of casual leave [1]."}),
    json.dumps(
        {
            "id": "q2",
            "answer": "The hotel limit is 180 US dollars per night [1].",
            "model": "alias A resolved",
            "input_tokens": 120,
            "output_tokens": 15,
        }
    ),
    json.dumps({"id": "q3", "answer": NO_EVIDENCE_ANSWER}),
]


def _import(loaded, tmp_path, lines, label="hosted model, alias A", batch="ext"):
    _, factory, _ = loaded
    prompts = _export(loaded, "traditional", tmp_path)
    answers = _write_answers(tmp_path / "answers.jsonl", lines)
    result = import_answers(
        prompts, answers, LexicalJudge(), factory, label=label, batch=batch, commit="def5678"
    )
    return result, factory, prompts


def test_import_scores_stores_and_labels_the_run(loaded, tmp_path):
    result, factory, prompts = _import(loaded, tmp_path, GOOD)
    assert result.target == "external:hosted-model-alias-a" == target_name("hosted model, alias A")
    assert not any(result.error_counts.values())
    with factory() as db:
        ((run, rows),) = batch_runs(db, "ext")
    assert run.strategy == "external:hosted-model-alias-a" and run.finished_at is not None
    meta = run.summary["meta"]
    assert meta["answered_by"] == "hosted model, alias A"
    assert meta["answer_source"] == "external answers file"
    assert len(meta["prompt_file_sha256"]) == 64 and meta["export_date"]
    assert meta["export_commit"] == "abc1234" and meta["reported_models"] == ["alias A resolved"]
    by_id = {r.question_id: r for r in rows}
    assert by_id["q1"].hit is True and by_id["q1"].recall == 1.0
    assert by_id["q1"].citation_correct is True
    # q3 declines with passages retrieved: the contract's own citation rule applies.
    assert by_id["q3"].citation_correct is False
    assert run.summary["refusals"] == 1 and run.summary["errors"] == 0
    assert by_id["q1"].correctness is not None
    assert by_id["q1"].details["input_tokens"] is None  # absent: null, never zero
    assert by_id["q1"].details["output_tokens"] is None
    assert by_id["q2"].details["input_tokens"] == 120
    assert run.summary["input_tokens"] == 120  # only the measured value is summed


def test_a_fabricated_quote_fails_the_citation_check(loaded, tmp_path):
    bad = json.dumps({"id": "q1", "answer": 'It says "50 days of casual leave" [1].'})
    result, factory, _ = _import(loaded, tmp_path, [bad, GOOD[1], GOOD[2]])
    with factory() as db:
        ((_, rows),) = batch_runs(db, "ext")
    assert rows[0].citation_correct is False


def test_named_counted_errors_and_no_zero_scores(loaded, tmp_path):
    lines = [
        GOOD[0],
        json.dumps({"id": "q1", "answer": "Another answer [1]."}),  # duplicate of q1
        json.dumps({"id": "zz", "answer": "No such question [1]."}),  # unknown id
        "this is not json",  # unreadable
        json.dumps({"id": "q2"}),  # invalid record: no answer
        json.dumps({"id": "q3", "answer": "x", "input_tokens": -4}),  # invalid tokens
    ]
    result, factory, _ = _import(loaded, tmp_path, lines)
    assert result.error_counts == {
        "unreadable_line": 1,
        "invalid_record": 2,
        "unknown_id": 1,
        "duplicate_id": 1,
        "missing_answer": 2,  # q2 and q3 never got a usable answer
    }
    assert {e["id"] for e in result.errors if e["kind"] == "missing_answer"} == {"q2", "q3"}
    with factory() as db:
        ((run, rows),) = batch_runs(db, "ext")
    assert len(rows) == 3 and run.summary["errors"] == 3
    for row in rows:  # no answer is scored, so nothing is a zero
        assert row.details["error"] and row.hit is None and row.correctness is None
        assert row.citation_correct is None and row.answer is None
    assert run.summary["metrics"]["hit"] is None
    assert run.summary["meta"]["import_errors"]["counts"]["duplicate_id"] == 1
    text = render_report([(run, rows)])
    assert "duplicate_id 1" in text and "missing_answer 2" in text


def test_duplicate_run_name_in_a_batch_is_refused(loaded, tmp_path):
    _import(loaded, tmp_path, GOOD)
    with pytest.raises(ValueError, match="already has a run"):
        _import(loaded, tmp_path, GOOD)
    _import(loaded, tmp_path, GOOD, label="Other model")  # another label is fine


def test_report_shows_provenance_next_to_the_numbers(loaded, tmp_path):
    _, factory, _ = _import(loaded, tmp_path, GOOD)
    with factory() as db:
        text = render_report(batch_runs(db, "ext"))
    assert "## Answers from outside RagFabric" in text
    assert "not answered by a RagFabric API call" in text
    assert "external:hosted-model-alias-a (external answers) |" in text
    assert "| hosted model, alias A | external answers file |" in text
    assert "alias A resolved" in text and "external answers file" in text


def test_report_of_a_normal_run_has_no_provenance_section():
    from ragfabric_core.evaluation.report import _external

    class R:
        summary = {"meta": {}}

    assert _external(R()) is None


def test_prompt_file_problems_are_named(tmp_path):
    p = tmp_path / "p.jsonl"
    p.write_text(json.dumps({"type": "question", "id": "a"}) + "\n")
    with pytest.raises(PromptFileError, match="first line must be the header"):
        read_prompts(p)
    p.write_text("not json\n")
    with pytest.raises(PromptFileError, match="line 1"):
        read_prompts(p)
    with pytest.raises(PromptFileError, match="not found"):
        read_prompts(tmp_path / "missing.jsonl")


def test_read_answers_ignores_only_blank_lines(tmp_path):
    p = tmp_path / "a.jsonl"
    p.write_text('\n{"id": "q1", "answer": "A [1]"}\n\n[1, 2]\n')
    got = read_answers(p, {"q1"})
    assert list(got.answers) == ["q1"]
    assert [e["kind"] for e in got.errors] == ["unreadable_line"]


def test_slug():
    assert slug("Hosted model, alias A") == "hosted-model-alias-a"
    with pytest.raises(ValueError):
        slug("  --  ")
