"""MultiHop-RAG: news articles and multi hop queries with reference answers.

The data lives on Hugging Face (``yixuantt/MultiHopRAG``); the GitHub
repository no longer carries it. Each evidence item names an article by URL
and quotes a fact verbatim from its body. The evidence phrase is the first 72
characters of that fact: RagFabric chunks overlap by 80 characters, so a phrase
of at most 80 always sits wholly inside at least one chunk, and a relevant
chunk cannot fail to match just because the fact straddles a boundary.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path

from ragfabric_core.evaluation.dataset import EvalQuestion, ExpectedSource, QuestionType
from ragfabric_core.evaluation.datasets.base import (
    DatasetSpec,
    ImportedDataset,
    RemoteFile,
    sample_corpus,
)

HF_BASE = "https://huggingface.co/datasets/yixuantt/MultiHopRAG/resolve/main"
EVIDENCE_CHARS = 72

TYPE_MAP = {
    "inference_query": QuestionType.MULTI_HOP,
    "comparison_query": QuestionType.COMPARISON,
    "temporal_query": QuestionType.MULTI_DOCUMENT,
    "null_query": QuestionType.AMBIGUOUS,
}


def slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:80].strip("-") or "article"


def _flat(text: str) -> str:
    return " ".join(text.split()).lower()


class MultiHopRag:
    spec = DatasetSpec(
        name="multihop-rag",
        title="MultiHop-RAG",
        source=f"Hugging Face dataset yixuantt/MultiHopRAG, {HF_BASE}",
        licence="ODC-BY (dataset README and Hugging Face card)",
        files=(
            RemoteFile(
                url=f"{HF_BASE}/corpus.json",
                filename="corpus.json",
                sha256="20b61b5ab84de84a927420c5d265b7ec8d859ae49980699958a787ade9e4d28f",
            ),
            RemoteFile(
                url=f"{HF_BASE}/MultiHopRAG.json",
                filename="MultiHopRAG.json",
                sha256="03cfb4926461f868684903aadc8024447bdda5bb3f6804741424cce338515bff",
            ),
        ),
    )

    def convert(
        self,
        paths: dict[str, Path],
        *,
        limit: int | None,
        questions: int | None,
        seed: int,
        fetch: Callable[[RemoteFile], Path] | None = None,
    ) -> ImportedDataset:
        articles = json.loads(paths["corpus.json"].read_text(encoding="utf-8"))
        queries = json.loads(paths["MultiHopRAG.json"].read_text(encoding="utf-8"))

        by_url: dict[str, dict] = {}
        filename: dict[str, str] = {}
        used: set[str] = set()
        for art in sorted(articles, key=lambda a: a["url"]):
            base = slug(art["title"])
            name, n = base + ".txt", 2
            while name in used:
                name, n = f"{base}-{n}.txt", n + 1
            used.add(name)
            by_url[art["url"]] = art
            filename[art["url"]] = name

        def document_text(art: dict) -> str:
            return (
                f"{art['title']}\nSource: {art.get('source') or ''}\n"
                f"Published: {art.get('published_at') or ''}\n\n{art['body']}"
            )

        relevant: dict[str, set[str]] = {}
        sources: dict[str, list[ExpectedSource]] = {}
        unknown_type = missing_article = evidence_not_found = 0
        for index, row in enumerate(queries):
            qid = f"{index:04d}"
            if row.get("question_type") not in TYPE_MAP:
                unknown_type += 1
                continue
            items: list[ExpectedSource] = []
            missing = False
            for ev in row.get("evidence_list") or []:
                art = by_url.get(ev.get("url"))
                if art is None:
                    missing = True
                    break
                phrase = (ev.get("fact") or "")[:EVIDENCE_CHARS].strip()
                if not phrase or _flat(phrase) not in _flat(document_text(art)):
                    evidence_not_found += 1
                    phrase = None
                src = ExpectedSource(document=filename[ev["url"]], evidence=phrase)
                if src not in items:
                    items.append(src)
            if missing:
                missing_article += 1
                continue
            relevant[qid] = {s.document for s in items}
            sources[qid] = items

        doc_for_name = {filename[u]: by_url[u] for u in by_url}
        qids, doc_names = sample_corpus(
            relevant, doc_for_name, limit=limit, questions=questions, seed=seed
        )
        chosen_rel = {d for q in qids for d in relevant[q]}
        built = []
        for q in qids:
            row = queries[int(q)]
            built.append(
                EvalQuestion(
                    id=f"multihop-rag-{q}",
                    question=row["query"],
                    expected_answer=row.get("answer") or "",
                    expected_sources=sources[q],
                    question_type=TYPE_MAP[row["question_type"]],
                    difficulty=row["question_type"],
                )
            )
        return ImportedDataset(
            documents={n: document_text(doc_for_name[n]).encode("utf-8") for n in doc_names},
            questions=built,
            description=(
                "MultiHop-RAG news corpus and queries, imported by ragfabric eval import. "
                f"Evidence phrases are the first {EVIDENCE_CHARS} characters of each fact."
            ),
            dropped=[
                {"what": "question", "count": unknown_type, "reason": "unknown question_type"},
                {
                    "what": "question",
                    "count": missing_article,
                    "reason": "evidence names an article missing from corpus.json",
                },
                {
                    "what": "evidence phrase",
                    "count": evidence_not_found,
                    "reason": "fact not found verbatim in its article; source kept, document only",
                },
                {
                    "what": "question",
                    "count": len(relevant) - len(qids),
                    "reason": "not sampled: over the question cap or its relevant documents "
                    "did not fit in half of the document budget",
                },
            ],
            counts={
                "relevant_documents": len(chosen_rel),
                "distractors": len(doc_names) - len(chosen_rel),
                "source_documents": len(articles),
                "source_questions": len(queries),
            },
        )


MULTIHOP_RAG = MultiHopRag()
