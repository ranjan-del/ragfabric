"""The shipped question set and the bring-your-own loader."""

import json
import re
from collections import Counter

import pytest

from ragfabric_core.evaluation.dataset import (
    QuestionType,
    content_hash,
    load_questions,
    shipped_corpus_paths,
    shipped_questions,
)


def _flat(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def test_the_shipped_set_has_three_questions_per_category():
    qs = shipped_questions()
    counts = Counter(q.question_type for q in qs.questions)
    assert counts == {t: 3 for t in QuestionType}
    assert len({q.id for q in qs.questions}) == len(qs.questions) == 24


def test_eight_documents_ship():
    names = sorted(p.name for p in shipped_corpus_paths())
    assert len(names) == 8
    assert "leave-policy-2025.md" in names and "leave-policy-2026.md" in names


def test_every_expected_source_and_evidence_exists_in_the_corpus():
    corpus = {p.name: _flat(p.read_text(encoding="utf-8")) for p in shipped_corpus_paths()}
    for q in shipped_questions().questions:
        assert q.expected_sources, q.id
        for source in q.expected_sources:
            assert source.document in corpus, (q.id, source.document)
            if source.evidence:
                assert _flat(source.evidence) in corpus[source.document], (q.id, source.evidence)


def _write(tmp_path, questions):
    path = tmp_path / "q.json"
    path.write_text(json.dumps({"version": 1, "name": "mine", "questions": questions}))
    return path


def _q(**over):
    q = {
        "id": "a",
        "question": "What?",
        "expected_answer": "That.",
        "expected_sources": [{"document": "x.md"}],
        "question_type": "simple_factual",
        "difficulty": "easy",
    }
    q.update(over)
    return q


def test_a_bring_your_own_file_loads(tmp_path):
    qs = load_questions(_write(tmp_path, [_q()]))
    assert qs.name == "mine" and qs.questions[0].expected_sources[0].evidence is None


def test_an_unknown_question_type_is_named(tmp_path):
    with pytest.raises(ValueError, match="question_type"):
        load_questions(_write(tmp_path, [_q(question_type="trivia")]))


def test_duplicate_ids_are_refused(tmp_path):
    with pytest.raises(ValueError, match="duplicate question id 'a'"):
        load_questions(_write(tmp_path, [_q(), _q()]))


def test_an_unknown_key_is_refused(tmp_path):
    with pytest.raises(ValueError, match="expected_sorces"):
        load_questions(_write(tmp_path, [_q(expected_sorces=[])]))


def test_a_missing_file_says_so(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        load_questions(tmp_path / "nope.json")


def test_content_hash_changes_with_content(tmp_path):
    a = _write(tmp_path, [_q()])
    h1 = content_hash([a])
    a.write_text(a.read_text().replace("What?", "Why?"))
    assert content_hash([a]) != h1
    assert len(h1) == 12
