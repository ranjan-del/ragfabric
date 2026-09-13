"""Every claimed upload format, tested against a real generated file.

MEMORY.md advertises PDF, DOCX, PPTX, TXT and CSV. A parser that is merely
imported proves nothing, so each test here builds genuine bytes for the format
(see ``ragfabric_core/testing/fixtures.py``), runs them through the real parser, and then repeats
the exercise through the upload endpoint so the round trip to a queryable,
citable chunk is covered too.
"""

from __future__ import annotations

import pytest

from ragfabric_core.ingest.parser import PAGE_BREAK, SUPPORTED_FORMATS, parse
from ragfabric_core.ingest.pipeline import EMPTY_TEXT_NOTE
from ragfabric_core.testing.fixtures import make_csv, make_docx, make_pdf, make_pptx, make_txt

PDF_PAGES = [
    "Quarterly revenue grew fifteen percent in the cloud segment.",
    "Headcount increased by forty engineers across three offices.",
]
DOCX_PARAGRAPHS = [
    "Remote Work Policy.",
    "Employees may work remotely three days each week with manager approval.",
]
DOCX_TABLE = [["Region", "Remote days"], ["EMEA", "3"], ["APAC", "2"]]
PPTX_SLIDES = [
    ("Product Roadmap", "Ship the semantic search API in the third quarter."),
    ("Known Risks", "Vector index rebuild time grows with corpus size."),
]


def _upload(client, headers, filename, data, content_type="application/octet-stream"):
    return client.post(
        "/api/documents/upload",
        files={"file": (filename, data, content_type)},
        headers=headers,
    )


# --- parser-level ------------------------------------------------------------


def test_parse_pdf_extracts_text_and_page_breaks():
    text = parse("report.pdf", make_pdf(PDF_PAGES))
    assert "Quarterly revenue" in text
    assert "Headcount increased" in text
    # Page boundaries must survive, otherwise citations cannot report a page.
    assert text.count(PAGE_BREAK) == 1


def test_parse_docx_extracts_paragraphs_and_table_cells():
    text = parse("policy.docx", make_docx(DOCX_PARAGRAPHS, table=DOCX_TABLE))
    assert "Remote Work Policy." in text
    assert "work remotely three days" in text
    # Table rows are flattened to pipe-separated lines, not silently dropped.
    assert "EMEA | 3" in text


def test_parse_pptx_extracts_every_slide_with_breaks():
    text = parse("deck.pptx", make_pptx(PPTX_SLIDES))
    assert "Product Roadmap" in text
    assert "semantic search API" in text
    assert "Known Risks" in text
    assert text.count(PAGE_BREAK) == 1


def test_parse_txt_and_csv():
    assert parse("notes.txt", make_txt("hello world")) == "hello world"
    csv_text = parse("staff.csv", make_csv(["name", "role"], [["Ada", "engineer"]]))
    assert "name: Ada" in csv_text
    assert "role: engineer" in csv_text


def test_parse_csv_handles_quoted_commas():
    # The stdlib csv reader is used precisely so an embedded comma is not
    # mistaken for a column break.
    data = make_csv(["name", "title"], [["Ada", "Engineer, Systems"]])
    text = parse("staff.csv", data)
    assert "title: Engineer, Systems" in text


def test_parse_rejects_unknown_extension():
    with pytest.raises(ValueError, match="Unsupported format"):
        parse("image.png", b"\x89PNG")


# --- through the API ---------------------------------------------------------


@pytest.mark.parametrize(
    ("filename", "data_factory", "needle"),
    [
        ("report.pdf", lambda: make_pdf(PDF_PAGES), "cloud segment"),
        (
            "policy.docx",
            lambda: make_docx(DOCX_PARAGRAPHS, table=DOCX_TABLE),
            "remotely three days",
        ),
        ("deck.pptx", lambda: make_pptx(PPTX_SLIDES), "semantic search API"),
        ("notes.txt", lambda: make_txt("The wifi password rotates every quarter."), "wifi"),
        (
            "staff.csv",
            lambda: make_csv(["name", "role"], [["Ada", "engineer"]]),
            "engineer",
        ),
    ],
)
def test_every_supported_format_ingests_and_becomes_searchable(
    client, auth_headers, filename, data_factory, needle
):
    response = _upload(client, auth_headers, filename, data_factory())
    assert response.status_code == 201, response.text
    document = response.json()
    assert document["status"] == "ready", document["error"]
    assert document["num_chunks"] >= 1
    assert document["format"] == filename.rsplit(".", 1)[-1]

    # The indexed content is reachable by search, which is the real proof the
    # parse produced usable text rather than an empty string.
    search = client.post(
        "/api/search/semantic",
        json={"query": needle, "top_k": 5},
        headers=auth_headers,
    )
    assert search.status_code == 200
    results = search.json()["results"]
    assert results, f"nothing retrievable for {filename}"
    assert any(r["filename"] == filename for r in results)


def test_supported_formats_constant_matches_the_tested_set():
    # Guards against advertising a format in the API error message that no test
    # (and possibly no parser) actually covers.
    assert set(SUPPORTED_FORMATS) == {"pdf", "docx", "pptx", "txt", "csv"}


def test_pdf_page_numbers_reach_the_citation(client, auth_headers):
    _upload(client, auth_headers, "report.pdf", make_pdf(PDF_PAGES))
    answer = client.post(
        "/api/search/query",
        json={"query": "how much did headcount increase"},
        headers=auth_headers,
    ).json()
    citation = answer["citations"][0]
    assert citation["filename"] == "report.pdf"
    # The headcount sentence is on the second PDF page.
    assert citation["page"] == 2


def test_text_free_file_is_flagged_rather_than_silently_empty(client, auth_headers):
    # A PDF with a page that draws no text: parses fine, indexes nothing.
    response = _upload(client, auth_headers, "blank.pdf", make_pdf([" "]))
    assert response.status_code == 201
    document = response.json()
    assert document["status"] == "ready"
    assert document["num_chunks"] == 0
    assert document["error"] == EMPTY_TEXT_NOTE


def test_corrupt_file_of_a_supported_type_fails_cleanly(client, auth_headers):
    # Right extension, garbage bytes. The row must exist and say why it failed
    # rather than 500-ing the request.
    response = _upload(client, auth_headers, "broken.docx", b"not a zip archive")
    assert response.status_code == 201
    document = response.json()
    assert document["status"] == "failed"
    assert document["error"]
    assert document["num_chunks"] == 0
