from ragfabric_core.ingest.chunk import chunk_text
from ragfabric_core.ingest.clean import clean_text, detect_sections, document_type_for
from ragfabric_core.ingest.parser import PAGE_BREAK


def test_clean_joins_hyphenated_line_breaks_and_collapses_whitespace():
    raw = "The employ-\nee handbook   covers\tleave.\n\n\n\nNext paragraph."
    assert clean_text(raw) == "The employee handbook covers leave.\n\nNext paragraph."


def test_clean_removes_lines_repeated_on_three_or_more_pages():
    page = "ACME Corp Confidential\nBody {n}\nPage {n} of 3"
    raw = PAGE_BREAK.join(page.format(n=n) for n in (1, 2, 3))
    cleaned = clean_text(raw)
    assert "ACME Corp Confidential" not in cleaned
    assert "Body 2" in cleaned
    assert cleaned.count(PAGE_BREAK) == 2


def test_clean_keeps_a_line_that_repeats_on_only_two_pages():
    raw = PAGE_BREAK.join(["Header\nA", "Header\nB", "Other\nC"])
    assert clean_text(raw).count("Header") == 2


def test_detect_sections_finds_numbered_caps_and_markdown_headings():
    text = (
        "1. Introduction\nsome text\nLEAVE POLICY\nmore text\n## Carry forward\nrules\nplain line"
    )
    found = detect_sections(text)
    assert [title for _, title in found] == ["1. Introduction", "LEAVE POLICY", "Carry forward"]
    assert found[0][0] == 0 and text[found[1][0] :].startswith("LEAVE POLICY")


def test_detect_sections_ignores_long_or_sentence_like_lines():
    text = "This is a long sentence that ends with a period and is not a heading.\nOK"
    assert detect_sections(text) == []


def test_chunk_text_attaches_the_nearest_preceding_section_when_asked():
    text = "1. Intro\n" + ("a " * 50) + "\n2. Leave\n" + ("b " * 50)
    chunks = chunk_text(text, chunk_size=60, overlap=0, sections=True)
    assert chunks[0]["section"] == "1. Intro"
    assert chunks[-1]["section"] == "2. Leave"
    assert all("section" in c for c in chunks)


def test_chunk_text_default_does_not_add_section_key():
    assert "section" not in chunk_text("hello world")[0]


def test_document_type_mapping():
    assert document_type_for("pdf") == "document"
    assert document_type_for("pptx") == "presentation"
    assert document_type_for("csv") == "table"
    assert document_type_for("txt") == "text" and document_type_for("md") == "text"
    assert document_type_for("xyz") == "other"
