"""Unit tests for the offline ingestion + answer-assembly building blocks.

These exercise the deterministic core (parse, chunk, embed, answer assembly)
without a database or the API layer. Several tests below need a plausible
"retrieved chunks" list to feed ``build_answer``; ``_retrieve`` produces one by
embedding the corpus and the query with the real ``HashingEmbedder`` and
ranking by cosine similarity, which is exactly what the deleted
``InMemoryVectorStore``/``Retriever`` pair did and nothing more, so the ranking
these tests observe is unchanged even though the deleted classes are gone
(Task 13).
"""

from __future__ import annotations

import re

import numpy as np

from ragfabric_core.generate import llm
from ragfabric_core.generate.answer import _confidence, _find_term_spans, build_answer
from ragfabric_core.ingest.chunk import chunk_text
from ragfabric_core.ingest.embed import HashingEmbedder, content_tokens, tokenize
from ragfabric_core.ingest.parser import PAGE_BREAK, parse


def _retrieve(docs: list[str], query: str, embedder: HashingEmbedder, top_k: int) -> list[dict]:
    """Rank ``docs`` by cosine similarity to ``query`` with the real embedder.

    Unit-normalised vectors make dot product equal cosine similarity, the same
    scoring the deleted ``InMemoryVectorStore.search`` used.
    """
    vectors = embedder.embed(docs)
    query_vector = embedder.embed_one(query)
    scores = vectors @ query_vector
    order = sorted(range(len(docs)), key=lambda i: (-float(scores[i]), i))[:top_k]
    return [
        {
            "chunk_id": i,
            "document_id": 1,
            "collection_id": None,
            "filename": "kb.txt",
            "page": 1,
            "chunk_index": i,
            "text": docs[i],
            "score": float(scores[i]),
        }
        for i in order
    ]


def test_parse_txt_and_csv():
    assert parse("notes.txt", b"hello world") == "hello world"
    csv_text = parse("data.csv", b"name,role\nAda,engineer\n")
    assert "name: Ada" in csv_text
    assert "role: engineer" in csv_text


def test_chunk_respects_page_breaks_and_overlap():
    text = f"page one content{PAGE_BREAK}page two content"
    chunks = chunk_text(text, chunk_size=50, overlap=10)
    pages = {c["page"] for c in chunks}
    assert pages == {1, 2}
    assert all(c["text"] for c in chunks)


def test_embedder_is_deterministic_and_normalized():
    embedder = HashingEmbedder(dim=64)
    a = embedder.embed_one("the quick brown fox")
    b = embedder.embed_one("the quick brown fox")
    assert np.allclose(a, b)  # reproducible across calls
    assert abs(float(np.linalg.norm(a)) - 1.0) < 1e-5  # unit length


def test_build_answer_produces_citations_confidence_and_highlights():
    embedder = HashingEmbedder(dim=256)
    docs = [
        "The security policy requires multi factor authentication for all admins.",
        "Coffee is available on every floor of the building.",
    ]
    retrieved = _retrieve(docs, "What authentication is required for admins?", embedder, top_k=2)
    result = build_answer("What authentication is required for admins?", retrieved)

    assert result["answer"]
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["citations"]
    assert result["citations"][0]["marker"] == "[1]"
    assert result["source_document"]["filename"] == "kb.txt"
    # At least one query term should be highlighted in the top chunk.
    assert any(h["term"] in ("authentication", "admins", "required") for h in result["highlights"])


def test_build_answer_handles_no_results():
    result = build_answer("anything", [])
    assert result["confidence"] == 0.0
    assert result["citations"] == []
    assert result["source_document"] is None


# --- tokenisation ------------------------------------------------------------


def test_content_tokens_drop_stopwords_but_never_everything():
    assert content_tokens("What is the vacation policy?") == ["vacation", "policy"]
    # An all-stopword string falls back to the raw tokens, because a zero vector
    # would score 0.0 against every chunk and give the caller no ranking at all.
    assert content_tokens("what is it") == tokenize("what is it")


def test_stopwords_stop_off_topic_queries_from_scoring():
    embedder = HashingEmbedder(dim=256)
    docs = ["The vacation policy grants employees twenty paid leave days per year."]

    # Shares only function words with the corpus, so it must score ~0. With
    # stopwords left in, this query scored 0.228 against this same chunk.
    off_topic = _retrieve(docs, "what is the capital of france", embedder, top_k=1)
    on_topic = _retrieve(docs, "how many paid leave days", embedder, top_k=1)
    assert off_topic[0]["score"] == 0.0
    assert on_topic[0]["score"] > 0.2


# --- chunking ----------------------------------------------------------------


def test_chunk_spans_exactly_bound_the_stored_text():
    text = "  " + "alpha beta gamma delta. " * 60
    for chunk in chunk_text(text, chunk_size=100, overlap=20):
        page_text = text  # single page, no form feeds
        assert page_text[chunk["char_start"] : chunk["char_end"]] == chunk["text"]


def test_chunk_overlap_shares_context_between_neighbours():
    text = "".join(f"sentence {i}. " for i in range(60))
    chunks = chunk_text(text, chunk_size=120, overlap=40)
    assert len(chunks) > 2
    # Consecutive windows advance by (chunk_size - overlap), so each pair shares
    # text; that is what keeps a fact straddling a boundary intact somewhere.
    assert chunks[0]["char_end"] > chunks[1]["char_start"]


# --- answer, citations, confidence, highlights -------------------------------


def test_answer_text_is_lifted_from_the_cited_chunk():
    """The core citation contract: each [n] quote comes from chunk n."""
    embedder = HashingEmbedder(dim=256)
    docs = [
        "Expense reports must be submitted within thirty days of travel. "
        "Receipts above fifty euros require manager approval.",
        "The security policy requires multi factor authentication for admins. "
        "Passwords rotate every ninety days.",
    ]
    query = "when must expense reports be submitted"

    retrieved = _retrieve(docs, query, embedder, top_k=2)
    result = build_answer(query, retrieved)

    body = result["answer"].split(":", 1)[1]
    quotes = [q.strip() for q in re.split(r"\[\d+\]", body) if q.strip()]
    markers = [int(m) for m in re.findall(r"\[(\d+)\]", result["answer"])]
    assert quotes and len(quotes) == len(markers)

    for quote, marker in zip(quotes, markers, strict=True):
        source = result["citations"][marker - 1]
        chunk = next(c for c in retrieved if c["chunk_id"] == source["chunk_id"])
        # Whitespace is collapsed when rendering, so compare on collapsed text.
        assert quote in " ".join(chunk["text"].split())


def test_citations_record_which_sources_the_answer_used():
    embedder = HashingEmbedder(dim=256)
    docs = [
        "Expense reports must be submitted within thirty days of travel.",
        "The office cafeteria serves lunch between noon and two.",
        "Bicycle parking is available in the basement.",
        "Visitor badges must be returned at reception.",
    ]
    query = "when must expense reports be submitted"

    retrieved = _retrieve(docs, query, embedder, top_k=4)
    result = build_answer(query, retrieved)

    # Every retrieved chunk gets a citation entry, but "retrieved" and "used"
    # are different claims and the payload has to keep them apart.
    assert len(result["citations"]) == 4

    used = [c for c in result["citations"] if c["used"]]
    # Strictly fewer than the three top-ranked chunks are quoted. Only the
    # chunks whose best sentence actually covers a word of the question earn a
    # marker; cafeteria hours and bicycle parking are retrieved because the
    # index returns its best four, not because they answer anything.
    assert 0 < len(used) < 4
    assert result["citations"][0]["used"] is True

    for citation in used:
        assert citation["marker"] in result["answer"]
    for citation in result["citations"]:
        if not citation["used"]:
            assert citation["marker"] not in result["answer"]


def test_confidence_separates_answerable_from_unanswerable_questions():
    embedder = HashingEmbedder(dim=256)
    docs = [
        "Full time employees receive twenty five paid vacation days each year.",
        "Managers approve leave requests two weeks in advance.",
    ]
    good = "how many paid vacation days do employees receive"
    bad = "what is the airspeed velocity of an unladen swallow"
    good_conf = build_answer(good, _retrieve(docs, good, embedder, top_k=2))["confidence"]
    bad_conf = build_answer(bad, _retrieve(docs, bad, embedder, top_k=2))["confidence"]

    assert 0.0 <= bad_conf <= 1.0 and 0.0 <= good_conf <= 1.0
    assert good_conf > 0.5
    assert bad_conf == 0.0
    assert good_conf > bad_conf


def test_confidence_is_zero_without_retrieved_chunks():
    assert _confidence("anything at all", []) == 0.0


def test_highlights_respect_word_boundaries():
    # "cat" must not light up inside "category": a substring search would.
    spans = _find_term_spans("cat", "the category lists one cat and one dog")
    assert [s["start"] for s in spans] == [23]
    assert spans[0]["term"] == "cat"


def test_highlight_spans_index_into_the_text_they_describe():
    text = "Multi factor authentication is required for every administrator account."
    spans = _find_term_spans("which authentication do administrators need", text)
    assert spans
    for span in spans:
        assert text[span["start"] : span["end"]].lower() == span["term"]


def test_citation_highlights_are_relative_to_the_snippet():
    embedder = HashingEmbedder(dim=256)
    docs = ["The security policy requires multi factor authentication for admins."]
    query = "what authentication is required for admins"

    result = build_answer(query, _retrieve(docs, query, embedder, top_k=1))
    citation = result["citations"][0]
    assert citation["highlights"]
    for span in citation["highlights"]:
        assert citation["snippet"][span["start"] : span["end"]].lower() == span["term"]


def test_supporting_span_marks_the_sentence_the_answer_quoted():
    """The span must be the answer's own words, not a re-guess after the fact."""
    embedder = HashingEmbedder(dim=256)
    docs = [
        "Company Leave Policy. Full time employees receive twenty five paid "
        "vacation days each year. Coffee is available on every floor.",
    ]
    query = "how many paid vacation days do employees receive"

    result = build_answer(query, _retrieve(docs, query, embedder, top_k=1))
    citation = result["citations"][0]
    span = citation["supporting_span"]

    assert span is not None
    # The span indexes into the snippet the UI renders...
    assert citation["snippet"][span["start"] : span["end"]] == span["text"]
    # ...and the answer really does quote that text.
    assert " ".join(span["text"].split()) in " ".join(result["answer"].split())
    assert "twenty five paid vacation days" in span["text"]


def test_retrieved_but_unquoted_chunks_have_no_supporting_span():
    embedder = HashingEmbedder(dim=256)
    docs = [
        "Expense reports must be submitted within thirty days of travel.",
        "The office cafeteria serves lunch between noon and two.",
        "Bicycle parking is available in the basement.",
        "Visitor badges must be returned at reception.",
    ]
    query = "when must expense reports be submitted"

    result = build_answer(query, _retrieve(docs, query, embedder, top_k=4))

    # A chunk is quoted only if it cleared the relevance floors, so the two
    # flags have to agree in both directions: a span means it was used, and no
    # span means it was not. Anything else shows the reader a highlighted
    # "source" for a sentence the answer never took anything from.
    for citation in result["citations"]:
        assert (citation["supporting_span"] is not None) == citation["used"]

    # The chunk that genuinely answers the question is quoted.
    assert result["citations"][0]["used"] is True
    assert result["citations"][0]["supporting_span"] is not None

    # The fourth chunk is beyond the quoting window entirely.
    assert result["citations"][3]["used"] is False
    assert result["citations"][3]["supporting_span"] is None


def test_snippet_window_follows_the_supporting_sentence_into_a_long_chunk():
    """A 240 character prefix would cut away the sentence that answers.

    The filler below pushes the answering sentence past the snippet budget, so
    a fixed ``text[:240]`` window would show the reader a "source" that does
    not contain the quoted text at all.
    """
    filler = "This paragraph is preamble about office logistics. " * 8
    chunk_text_body = (
        filler + "Severity one incidents require a written postmortem within five working days."
    )
    assert len(filler) > 240  # the sentence really is out of reach of a prefix

    chunk = {
        "text": chunk_text_body,
        "chunk_id": 1,
        "document_id": 1,
        "filename": "runbook.txt",
        "page": 1,
        "score": 0.5,
    }
    query = "when is a postmortem required for severity one incidents"
    result = build_answer(query, [chunk])

    citation = result["citations"][0]
    span = citation["supporting_span"]
    assert span is not None
    assert "written postmortem" in span["text"]
    assert citation["snippet"][span["start"] : span["end"]] == span["text"]
    # The window was cut out of the middle, so it is marked as elided.
    assert citation["snippet"].startswith("...")


def test_answer_highlights_index_into_the_answer_text():
    embedder = HashingEmbedder(dim=256)
    docs = ["The security policy requires multi factor authentication for admins."]
    query = "what authentication is required for admins"

    result = build_answer(query, _retrieve(docs, query, embedder, top_k=1))
    assert result["highlights"]
    for span in result["highlights"]:
        assert result["answer"][span["start"] : span["end"]].lower() == span["term"]


# --- offline answer generation ------------------------------------------------


def test_sentence_spans_cover_punctuation_and_newlines():
    text = "First sentence. Second one!\nThird line"
    spans = llm.sentence_spans(text)
    assert [text[s:e] for s, e in spans] == [
        "First sentence.",
        "Second one!",
        "Third line",
    ]


def test_select_support_prefers_the_sentence_that_answers():
    chunk = {
        "text": (
            "Company Leave Policy. Full time employees receive twenty five paid "
            "vacation days each year. Coffee is available on every floor."
        )
    }
    support = llm.select_support("how many vacation days", [chunk])
    assert len(support) == 1
    assert "twenty five paid vacation days" in support[0]["text"]
    # Offsets must point back into the chunk exactly.
    assert chunk["text"][support[0]["start"] : support[0]["end"]] == support[0]["text"]


def test_generate_is_always_the_offline_extractive_answer():
    # generate() no longer has a vendor branch: a model-backed answer goes
    # through generate.cited and a configured LLMProvider instead, so this
    # module always returns the deterministic extractive answer.
    chunks = [{"text": "Passwords rotate every ninety days."}]
    answer = llm.generate("how often do passwords rotate", "[1] ...", chunks=chunks)
    assert "ninety days" in answer
    assert "[1]" in answer


def test_cited_markers_parses_the_answer_text():
    assert llm.cited_markers("Based on [1] and also [3].") == {1, 3}
    assert llm.cited_markers("no markers here") == set()


# --- Relevance floors on citation selection ---------------------------------
#
# "Top three chunks" used to be taken literally: rank decided what got quoted,
# so a chunk sharing no words at all with the question was still quoted and
# still marked used=true. These tests pin the floors that replaced that.


def test_an_irrelevant_chunk_is_not_quoted_just_for_being_in_the_top_three():
    chunks = [
        {"text": "Full time employees receive twenty five paid vacation days each year."},
        {"text": "Bicycle parking is available in the basement."},
        {"text": "The lobby fountain is cleaned on Tuesdays."},
    ]
    support = llm.select_support("how many vacation days do employees receive", chunks)

    assert len(support) == 1
    assert support[0]["marker"] == 1
    assert "vacation days" in support[0]["text"]


def test_a_chunk_the_retriever_scored_zero_is_never_quoted():
    """An explicit zero score is the retriever saying it found nothing."""
    chunks = [
        {"text": "Full time employees receive twenty five paid vacation days.", "score": 0.7},
        {"text": "Employees receive a welcome pack on their first day.", "score": 0.0},
    ]
    support = llm.select_support("how many vacation days do employees receive", chunks)

    assert [item["marker"] for item in support] == [1]


def test_a_missing_score_is_not_treated_as_a_zero_one():
    """Callers that assemble chunks themselves never set a score field."""
    chunks = [{"text": "Full time employees receive twenty five paid vacation days."}]
    assert len(llm.select_support("how many vacation days", chunks)) == 1


def test_markers_stay_aligned_with_the_retrieved_list_when_a_chunk_is_skipped():
    """Marker n must always mean "the nth retrieved chunk".

    Renumbering to close the gap would be the natural-looking thing to do and
    would silently point every citation after the skipped one at the wrong
    source document.
    """
    chunks = [
        {"text": "Bicycle parking is available in the basement."},
        {"text": "Full time employees receive twenty five paid vacation days."},
    ]
    support = llm.select_support("how many vacation days", chunks)

    assert [item["marker"] for item in support] == [2]


def test_a_question_nothing_answers_produces_no_citations_at_all():
    chunks = [
        {"text": "Bicycle parking is available in the basement."},
        {"text": "The lobby fountain is cleaned on Tuesdays."},
    ]
    question = "what is the airspeed velocity of an unladen swallow"

    assert llm.select_support(question, chunks) == []
    # And the answer says so rather than quoting something unrelated.
    assert "don't have enough information" in llm.extractive_answer(question, chunks)
