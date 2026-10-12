"""Importer core and BEIR: a hand written miniature zip, read through a fake fetcher.

Never the network: the fetcher copies bytes the test built itself.
"""

import hashlib
import io
import json
import zipfile
from pathlib import Path

import pytest

from ragfabric_core.evaluation.dataset import load_questions
from ragfabric_core.evaluation.datasets import DATASETS, import_dataset
from ragfabric_core.evaluation.datasets.base import (
    ChecksumError,
    RemoteFile,
    fetch_cached,
    safe_filename,
    sample_corpus,
)

DOCS = {
    f"d{i}": {"title": f"Title {i}", "text": f"Body of document {i} about topic {i}."}
    for i in range(1, 9)
}
QUERIES = {"q1": "What is topic 1?", "q2": "Tell me about topics 2 and 3.", "q3": "Topic 4?"}
QRELS = [
    ("q1", "d1", 1),
    ("q2", "d2", 1),
    ("q2", "d3", 2),
    ("q3", "d4", 1),
    ("q3", "d99", 1),
    ("q3", "d5", 0),
]


def mini_beir_zip(name="scifact") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(
            f"{name}/corpus.jsonl",
            "\n".join(json.dumps({"_id": k, **v}) for k, v in DOCS.items()) + "\n",
        )
        zf.writestr(
            f"{name}/queries.jsonl",
            "\n".join(
                json.dumps({"_id": k, "text": v, "metadata": {}}) for k, v in QUERIES.items()
            ),
        )
        zf.writestr(
            f"{name}/qrels/test.tsv",
            "query-id\tcorpus-id\tscore\n" + "".join(f"{q}\t{d}\t{s}\n" for q, d, s in QRELS),
        )
    return buf.getvalue()


class FakeFetcher:
    def __init__(self, payloads: dict[str, bytes]):
        self.payloads = payloads
        self.calls: list[str] = []

    def __call__(self, url: str, dest: Path) -> None:
        self.calls.append(url)
        dest.write_bytes(self.payloads[url])


def scifact_fetcher(data: bytes | None = None) -> FakeFetcher:
    spec = DATASETS["scifact"].spec
    return FakeFetcher({spec.files[0].url: data or mini_beir_zip()})


def run_import(tmp_path, out="out", fetcher=None, **kwargs):
    return import_dataset(
        "scifact",
        tmp_path / out,
        cache_dir=tmp_path / "cache",
        fetcher=fetcher or scifact_fetcher(),
        verify_checksums=False,
        **kwargs,
    )


def test_a_miniature_beir_zip_converts_to_a_valid_question_set(tmp_path):
    manifest = run_import(tmp_path)
    out = tmp_path / "out"
    qs = load_questions(out / "questions.json")
    assert {q.id for q in qs.questions} == {"scifact-q1", "scifact-q2", "scifact-q3"}
    q2 = next(q for q in qs.questions if q.id == "scifact-q2")
    assert q2.question == "Tell me about topics 2 and 3."
    assert [s.document for s in q2.expected_sources] == ["d2.txt", "d3.txt"]
    assert all(s.evidence is None for s in q2.expected_sources)
    assert q2.expected_answer == "" and q2.question_type == "simple_factual"
    assert sorted(p.name for p in (out / "corpus").iterdir()) == [f"d{i}.txt" for i in range(1, 9)]
    assert (out / "corpus" / "d1.txt").read_text() == "Title 1\n\nBody of document 1 about topic 1."
    assert manifest["counts"]["documents"] == 8 and manifest["counts"]["questions"] == 3


def test_qrels_with_score_zero_or_a_missing_document_are_dropped_and_reported(tmp_path):
    manifest = run_import(tmp_path)
    q3 = next(
        q for q in load_questions(tmp_path / "out/questions.json").questions if q.id == "scifact-q3"
    )
    assert [s.document for s in q3.expected_sources] == ["d4.txt"]
    reasons = {d["reason"]: d["count"] for d in manifest["dropped"]}
    assert reasons["judgement names a document missing from the corpus"] == 1
    assert reasons["relevance score is 0"] == 1


def test_the_manifest_records_licence_source_checksums_and_sampling(tmp_path):
    data = mini_beir_zip()
    manifest = run_import(tmp_path, fetcher=scifact_fetcher(data), limit=6, seed=3)
    on_disk = json.loads((tmp_path / "out/manifest.json").read_text())
    assert on_disk == manifest
    assert manifest["dataset"] == "scifact"
    assert manifest["licence"] == DATASETS["scifact"].spec.licence
    assert manifest["files"][0]["sha256"] == hashlib.sha256(data).hexdigest()
    assert manifest["files"][0]["url"].startswith("https://")
    assert manifest["sampling"] == {"limit": 6, "questions": None, "seed": 3}


def test_limit_counts_documents_and_relevant_documents_are_always_included(tmp_path):
    run_import(tmp_path, limit=6, seed=1)
    out = tmp_path / "out"
    corpus = {p.name for p in (out / "corpus").iterdir()}
    assert len(corpus) == 6
    for q in load_questions(out / "questions.json").questions:
        assert {s.document for s in q.expected_sources} <= corpus


def test_relevant_documents_never_take_more_than_half_the_budget(tmp_path):
    manifest = run_import(tmp_path, limit=4, seed=1)
    qs = load_questions(tmp_path / "out/questions.json").questions
    relevant = {s.document for q in qs for s in q.expected_sources}
    assert len(relevant) <= 2
    assert manifest["counts"]["relevant_documents"] == len(relevant)
    assert manifest["counts"]["distractors"] == 4 - len(relevant)


def test_sampling_is_deterministic_for_a_seed_and_differs_for_another():
    rel = {f"q{i}": {f"r{i}"} for i in range(30)}
    docs = sorted({f"r{i}" for i in range(30)} | {f"x{i}" for i in range(200)})
    a = sample_corpus(rel, docs, limit=20, questions=5, seed=7)
    b = sample_corpus(rel, docs, limit=20, questions=5, seed=7)
    c = sample_corpus(rel, docs, limit=20, questions=5, seed=8)
    assert a == b
    assert a != c
    qids, chosen = a
    assert len(qids) == 5 and len(chosen) == 20


def test_same_arguments_give_byte_identical_files(tmp_path):
    run_import(tmp_path, out="a", limit=6, seed=5)
    run_import(tmp_path, out="b", limit=6, seed=5)
    for name in ("questions.json", "manifest.json"):
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()


def test_questions_caps_the_question_count(tmp_path):
    manifest = run_import(tmp_path, questions=2, seed=1)
    assert len(load_questions(tmp_path / "out/questions.json").questions) == 2
    assert manifest["counts"]["questions"] == 2


def test_a_cached_file_is_reused_without_fetching_again(tmp_path):
    fetcher = scifact_fetcher()
    run_import(tmp_path, out="a", fetcher=fetcher)
    run_import(tmp_path, out="b", fetcher=fetcher)
    assert len(fetcher.calls) == 1


def test_a_checksum_mismatch_on_a_cached_file_re_downloads(tmp_path):
    good = b"the real bytes"
    remote = RemoteFile(
        url="https://example.org/f.zip", filename="f.zip", sha256=hashlib.sha256(good).hexdigest()
    )
    cache = tmp_path / "cache"
    (cache / "demo").mkdir(parents=True)
    (cache / "demo" / "f.zip").write_bytes(b"corrupted or truncated")
    fetcher = FakeFetcher({remote.url: good})
    path = fetch_cached("demo", remote, cache, fetcher)
    assert fetcher.calls == [remote.url]
    assert path.read_bytes() == good


def test_a_download_that_does_not_match_the_pinned_checksum_is_refused(tmp_path):
    remote = RemoteFile(url="https://example.org/f.zip", filename="f.zip", sha256="0" * 64)
    with pytest.raises(ChecksumError, match="f.zip"):
        fetch_cached("demo", remote, tmp_path, FakeFetcher({remote.url: b"other"}))
    assert not (tmp_path / "demo" / "f.zip").exists()


def test_an_unpinned_file_is_checked_against_its_first_download(tmp_path):
    remote = RemoteFile(url="https://example.org/g.json", filename="g.json", sha256=None)
    fetcher = FakeFetcher({remote.url: b"v1"})
    fetch_cached("demo", remote, tmp_path, fetcher)
    (tmp_path / "demo" / "g.json").write_bytes(b"tampered")
    fetch_cached("demo", remote, tmp_path, fetcher)
    assert len(fetcher.calls) == 2
    assert (tmp_path / "demo" / "g.json").read_bytes() == b"v1"


def test_every_beir_dataset_is_registered_with_a_licence_and_a_pinned_checksum():
    for name in ("scifact", "nfcorpus", "fiqa"):
        spec = DATASETS[name].spec
        assert spec.licence and spec.source
        assert all(f.sha256 and len(f.sha256) == 64 for f in spec.files)


def test_unknown_dataset_names_the_known_ones(tmp_path):
    with pytest.raises(ValueError, match="scifact"):
        import_dataset("nope", tmp_path / "o", cache_dir=tmp_path)


@pytest.mark.parametrize("raw,expected", [("MED-10", "MED-10"), ("a/b c", "a_b_c"), ("..", "_..")])
def test_safe_filename(raw, expected):
    assert safe_filename(raw) == expected
