"""The hand-labelled faithfulness set used to measure the judge (ADR 0004).

The labels are a measuring instrument, so they are checked like one: every passage is text from the
shipped synthetic corpus, every case carries a one-line reason, and a supported answer scripted as
supported is scored 1, which the v1 judge did not do for the same triples.
"""

import json
import re
from pathlib import Path

import pytest

from ragfabric_core.evaluation.dataset import shipped_corpus_paths
from ragfabric_core.evaluation.judge import LLMJudge
from ragfabric_core.evaluation.target import ContextItem
from ragfabric_core.providers.offline import ScriptedLLMProvider

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "faithfulness_labels.json").read_text())


def _flat(text):
    return re.sub(r"\s+", " ", text).strip().lower()


def test_the_set_has_at_least_thirty_cases_and_both_labels():
    labels = [c["faithful"] for c in FIXTURE["cases"]]
    assert len(labels) >= 30
    assert True in labels and False in labels
    assert all(isinstance(v, bool) for v in labels)


def test_every_passage_comes_from_the_shipped_corpus():
    corpus = {p.name: _flat(p.read_text()) for p in shipped_corpus_paths()}
    for pid, passage in FIXTURE["passages"].items():
        assert _flat(passage["text"]) in corpus[passage["document"]], pid


def test_every_case_has_a_reason_and_known_passages():
    ids = set()
    for case in FIXTURE["cases"]:
        assert case["id"] not in ids
        ids.add(case["id"])
        assert case["reason"].strip() and case["answer"].strip()
        assert case["passages"] and set(case["passages"]) <= set(FIXTURE["passages"])


@pytest.mark.parametrize("case", [c for c in FIXTURE["cases"] if c["faithful"]][:5])
def test_a_supported_answer_with_every_claim_supported_scores_one(case):
    reply = json.dumps({"claims": [{"claim": case["answer"], "supported": True}]})
    llm = ScriptedLLMProvider(responses=[reply], model="judge")
    contexts = [
        ContextItem(
            rank=i,
            document=FIXTURE["passages"][p]["document"],
            text=FIXTURE["passages"][p]["text"],
        )
        for i, p in enumerate(case["passages"], start=1)
    ]
    s = LLMJudge(llm, model=None, metrics=("faithfulness",)).score(
        case["question"], "", case["answer"], contexts
    )
    assert s.faithfulness == 1.0
