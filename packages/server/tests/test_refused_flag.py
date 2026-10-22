"""The API says whether an answer is a refusal, so clients never read it off the wording."""

from __future__ import annotations

import json

from ragfabric_core.generate.answer import build_answer
from ragfabric_core.generate.cited import NO_EVIDENCE_ANSWER
from ragfabric_core.generate.contract import is_refusal


def _done(raw: str) -> dict:
    done = None
    for block in raw.strip().split("\n\n"):
        lines = block.splitlines()
        if lines and lines[0] == "event: done":
            done = json.loads(lines[1].split(":", 1)[1])
    assert done is not None, raw
    return done


def _empty_collection(client, headers) -> int:
    return client.post("/api/collections", json={"name": "empty"}, headers=headers).json()["id"]


def test_a_refusal_is_flagged_by_the_same_rule_the_contract_uses():
    assert is_refusal(NO_EVIDENCE_ANSWER)
    assert is_refusal(f"{NO_EVIDENCE_ANSWER} The search stopped early (budget).")
    # An answer that cites a passage is not a refusal, even if it admits a gap.
    assert not is_refusal("The limit is 180 US dollars [1]. I could not find the date.")
    assert not is_refusal("The hotel limit is 180 US dollars per night.")
    assert not is_refusal("")
    assert not is_refusal(None)


def test_build_answer_flags_no_evidence_and_nothing_retrieved():
    rows = [{"chunk_id": 1, "document_id": 1, "text": "Staff get 24 days of leave.", "score": 0.9}]
    assert build_answer("leave", rows, answer_text=NO_EVIDENCE_ANSWER)["refused"] is True
    assert build_answer("leave", rows, answer_text="24 days [1].")["refused"] is False
    # With nothing retrieved the answer is a refusal whatever it says.
    assert build_answer("leave", [], answer_text="I do not know.")["refused"] is True


def test_a_non_streamed_answer_carries_refused(client, admin_headers, ingested_doc):
    empty = _empty_collection(client, admin_headers)
    refused = client.post(
        "/api/ask",
        json={"query": "annual leave", "stream": False, "collection_id": empty},
        headers=admin_headers,
    ).json()
    answered = client.post(
        "/api/ask", json={"query": "annual leave", "stream": False}, headers=admin_headers
    ).json()

    assert refused["refused"] is True and refused["citations"] == []
    assert answered["refused"] is False and answered["citations"]


def test_the_search_query_response_carries_refused(client, admin_headers, ingested_doc):
    empty = _empty_collection(client, admin_headers)
    refused = client.post(
        "/api/search/query",
        json={"query": "annual leave", "collection_id": empty},
        headers=admin_headers,
    ).json()
    answered = client.post(
        "/api/search/query", json={"query": "annual leave"}, headers=admin_headers
    ).json()

    assert refused["refused"] is True
    assert answered["refused"] is False


def test_the_streamed_done_event_carries_refused(client, admin_headers, ingested_doc):
    empty = _empty_collection(client, admin_headers)
    with client.stream(
        "POST",
        "/api/ask",
        json={"query": "annual leave", "stream": True, "collection_id": empty},
        headers=admin_headers,
    ) as res:
        refused = _done("".join(res.iter_text()))
    with client.stream(
        "POST", "/api/ask", json={"query": "annual leave", "stream": True}, headers=admin_headers
    ) as res:
        answered = _done("".join(res.iter_text()))

    assert refused["refused"] is True
    assert answered["refused"] is False


def test_the_field_is_optional_for_a_client_that_ignores_it(client, admin_headers, ingested_doc):
    body = client.post(
        "/api/ask", json={"query": "annual leave", "stream": False}, headers=admin_headers
    ).json()
    # Every field the response had before is still there next to the new one.
    assert {"question", "answer", "confidence", "citations", "highlights", "usage"} <= set(body)
