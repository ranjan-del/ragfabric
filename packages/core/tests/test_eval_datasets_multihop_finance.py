"""MultiHop-RAG and FinanceBench importers on hand written miniatures, through a fake fetcher."""

import json
from pathlib import Path

from ragfabric_core.evaluation.dataset import load_questions
from ragfabric_core.evaluation.datasets import DATASETS, import_dataset

LONG_FACT = (
    "Before his fall, Bankman-Fried made himself out to be the Good Boy of crypto, "
    "the trustworthy face of a sometimes-shady industry."
)
ARTICLES = [
    {
        "title": "The FTX trial is bigger than SBF",
        "author": "A",
        "source": "The Verge",
        "published_at": "2023-09-28T12:00:00+00:00",
        "category": "technology",
        "url": "https://example.org/ftx",
        "body": "Opening words. " + LONG_FACT + " Closing words.",
    },
    {
        "title": "SBF's trial starts soon",
        "author": "B",
        "source": "TechCrunch",
        "published_at": "2023-09-29T12:00:00+00:00",
        "category": "technology",
        "url": "https://example.org/sbf",
        "body": "Prosecutors say he committed fraud for personal gain.",
    },
    {
        "title": "Rates held steady",
        "author": "C",
        "source": "BBC",
        "published_at": "2023-10-01T00:00:00+00:00",
        "category": "business",
        "url": "https://example.org/rates",
        "body": "The bank held rates at 5.25 percent.",
    },
    {
        "title": "Unrelated sports story",
        "author": "D",
        "source": "ESPN",
        "published_at": "2023-10-02T00:00:00+00:00",
        "category": "sports",
        "url": "https://example.org/sports",
        "body": "A team won a match.",
    },
]


def _ev(url, fact):
    return {"title": "t", "url": url, "source": "s", "fact": fact}


QUERIES = [
    {
        "query": "Who is accused of fraud for personal gain?",
        "answer": "Sam Bankman-Fried",
        "question_type": "inference_query",
        "evidence_list": [
            _ev("https://example.org/ftx", LONG_FACT),
            _ev("https://example.org/sbf", "Prosecutors say he committed fraud for personal gain."),
        ],
    },
    {
        "query": "Did The Verge and TechCrunch agree?",
        "answer": "Yes",
        "question_type": "comparison_query",
        "evidence_list": [_ev("https://example.org/sbf", "committed fraud")],
    },
    {
        "query": "Was the rates story published after the FTX story?",
        "answer": "Yes",
        "question_type": "temporal_query",
        "evidence_list": [_ev("https://example.org/rates", "held rates at 5.25 percent")],
    },
    {
        "query": "Which character won a posthumous award?",
        "answer": "Insufficient information.",
        "question_type": "null_query",
        "evidence_list": [],
    },
]


class FakeFetcher:
    def __init__(self, payloads):
        self.payloads = payloads
        self.calls = []

    def __call__(self, url, dest: Path):
        self.calls.append(url)
        dest.write_bytes(self.payloads[url])


def multihop_payloads():
    spec = DATASETS["multihop-rag"].spec
    by_name = {f.filename: f.url for f in spec.files}
    return {
        by_name["corpus.json"]: json.dumps(ARTICLES).encode(),
        by_name["MultiHopRAG.json"]: json.dumps(QUERIES).encode(),
    }


def import_multihop(tmp_path, **kwargs):
    return import_dataset(
        "multihop-rag",
        tmp_path / "out",
        cache_dir=tmp_path / "cache",
        fetcher=FakeFetcher(multihop_payloads()),
        verify_checksums=False,
        **kwargs,
    )


def by_question(tmp_path):
    return {q.question: q for q in load_questions(tmp_path / "out/questions.json").questions}


def test_multihop_converts_with_the_type_mapping(tmp_path):
    import_multihop(tmp_path)
    qs = by_question(tmp_path)
    assert qs["Who is accused of fraud for personal gain?"].question_type == "multi_hop"
    assert qs["Did The Verge and TechCrunch agree?"].question_type == "comparison"
    assert qs["Was the rates story published after the FTX story?"].question_type == (
        "multi_document"
    )
    null = qs["Which character won a posthumous award?"]
    assert null.question_type == "ambiguous" and null.expected_sources == []
    assert null.expected_answer == "Insufficient information."
    assert qs["Who is accused of fraud for personal gain?"].expected_answer == "Sam Bankman-Fried"


def test_multihop_documents_carry_title_source_date_and_body(tmp_path):
    import_multihop(tmp_path)
    corpus = tmp_path / "out" / "corpus"
    names = sorted(p.name for p in corpus.iterdir())
    assert "the-ftx-trial-is-bigger-than-sbf.txt" in names and len(names) == 4
    text = (corpus / "the-ftx-trial-is-bigger-than-sbf.txt").read_text()
    assert text.splitlines()[0] == "The FTX trial is bigger than SBF"
    assert "The Verge" in text and "2023-09-28" in text and LONG_FACT in text


def test_multihop_evidence_is_at_most_72_characters_and_present_in_its_document(tmp_path):
    import_multihop(tmp_path)
    corpus = tmp_path / "out" / "corpus"
    for q in load_questions(tmp_path / "out/questions.json").questions:
        for source in q.expected_sources:
            assert source.evidence and len(source.evidence) <= 72
            body = " ".join((corpus / source.document).read_text().split()).lower()
            assert " ".join(source.evidence.split()).lower() in body
    first = by_question(tmp_path)["Who is accused of fraud for personal gain?"]
    assert first.expected_sources[0].evidence == LONG_FACT[:72].strip()
    assert [s.document for s in first.expected_sources] == [
        "the-ftx-trial-is-bigger-than-sbf.txt",
        "sbf-s-trial-starts-soon.txt",
    ]


def test_multihop_limit_keeps_every_evidence_article(tmp_path):
    import_multihop(tmp_path, limit=2, seed=4)
    corpus = {p.name for p in (tmp_path / "out" / "corpus").iterdir()}
    assert len(corpus) == 2
    for q in load_questions(tmp_path / "out/questions.json").questions:
        assert {s.document for s in q.expected_sources} <= corpus


# A minimal one page PDF, written by hand: enough for the importer, which copies bytes.
TINY_PDF = (
    b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
    b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
    b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 200 200]>>endobj\n"
    b"trailer<</Root 1 0 R>>\n%%EOF\n"
)
FB_QUESTIONS = [
    {
        "financebench_id": "financebench_id_00001",
        "doc_name": "ACME_2018_10K",
        "question_reasoning": "Numerical reasoning",
        "question": "What is ACME's FY2018 capex to revenue ratio?",
        "answer": "0.05",
        "evidence": [{"evidence_text": "a whole page", "evidence_page_num": 3}],
    },
    {
        "financebench_id": "financebench_id_00002",
        "doc_name": "ACME_2019_10K",
        "question_reasoning": "Information extraction",
        "question": "Who audits ACME?",
        "answer": "A big firm",
        "evidence": [],
    },
]
FB_DOCS = [
    {"doc_name": n, "doc_type": "10k", "doc_period": 2018}
    for n in ("ACME_2018_10K", "ACME_2019_10K", "OTHER_2020_10K")
]


def finance_payloads():
    spec = DATASETS["financebench"].spec
    by_name = {f.filename: f.url for f in spec.files}
    payloads = {
        by_name["financebench_open_source.jsonl"]: "\n".join(
            json.dumps(r) for r in FB_QUESTIONS
        ).encode(),
        by_name["financebench_document_information.jsonl"]: "\n".join(
            json.dumps(r) for r in FB_DOCS
        ).encode(),
    }
    from ragfabric_core.evaluation.datasets.financebench import pdf_url

    for doc in FB_DOCS:
        payloads[pdf_url(doc["doc_name"])] = TINY_PDF + doc["doc_name"].encode()
    return payloads


def test_financebench_converts_and_names_the_pdf(tmp_path):
    fetcher = FakeFetcher(finance_payloads())
    manifest = import_dataset(
        "financebench",
        tmp_path / "out",
        cache_dir=tmp_path / "cache",
        fetcher=fetcher,
        verify_checksums=False,
    )
    qs = {q.id: q for q in load_questions(tmp_path / "out/questions.json").questions}
    q1 = qs["financebench-financebench_id_00001"]
    assert [s.document for s in q1.expected_sources] == ["ACME_2018_10K.pdf"]
    assert q1.expected_sources[0].evidence is None
    assert q1.question_type == "complex_reasoning" and q1.expected_answer == "0.05"
    assert qs["financebench-financebench_id_00002"].question_type == "simple_factual"
    pdf = tmp_path / "out" / "corpus" / "ACME_2018_10K.pdf"
    assert pdf.read_bytes().startswith(b"%PDF")
    assert any(f["filename"] == "ACME_2018_10K.pdf" for f in manifest["files"])
    assert "CC BY-NC" in manifest["licence"]


def test_financebench_limit_fetches_only_the_sampled_pdfs(tmp_path):
    fetcher = FakeFetcher(finance_payloads())
    import_dataset(
        "financebench",
        tmp_path / "out",
        cache_dir=tmp_path / "cache",
        fetcher=fetcher,
        verify_checksums=False,
        limit=2,
        questions=1,
        seed=2,
    )
    pdf_calls = [u for u in fetcher.calls if u.endswith(".pdf")]
    assert len(pdf_calls) == 2
    assert len(list((tmp_path / "out" / "corpus").iterdir())) == 2
