"""FinanceBench open sample: 150 questions over SEC filings, with reference answers.

Questions and document list come from ``patronus-ai/financebench`` on GitHub;
each filing is fetched as its original PDF, only for the documents the sample
keeps. Evidence in FinanceBench is a whole page, so expected sources name the
document only. A question whose reasoning label mentions numerical reasoning
needs calculation and is typed ``complex_reasoning``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from ragfabric_core.evaluation.dataset import EvalQuestion, ExpectedSource, QuestionType
from ragfabric_core.evaluation.datasets.base import (
    DatasetSpec,
    ImportedDataset,
    RemoteFile,
    safe_filename,
    sample_corpus,
    sha256_of,
)

GH_BASE = "https://raw.githubusercontent.com/patronus-ai/financebench/main"


def pdf_url(doc_name: str) -> str:
    return f"{GH_BASE}/pdfs/{doc_name}.pdf"


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


class FinanceBench:
    spec = DatasetSpec(
        name="financebench",
        title="FinanceBench (open sample)",
        source=f"GitHub patronus-ai/financebench, {GH_BASE}/data and {GH_BASE}/pdfs",
        licence="CC BY-NC 4.0 (Hugging Face card PatronusAI/financebench); "
        "the GitHub repository has no licence file",
        files=(
            RemoteFile(
                url=f"{GH_BASE}/data/financebench_open_source.jsonl",
                filename="financebench_open_source.jsonl",
                sha256="a5a2aa673e573e55675fc3c0f9aa38c1cf59d2abc91edb077534f71f10a71877",
            ),
            RemoteFile(
                url=f"{GH_BASE}/data/financebench_document_information.jsonl",
                filename="financebench_document_information.jsonl",
                sha256="1c69127783879de8cdadb159d2181f39bc3123b8e0ebf74031c3969d69189575",
            ),
        ),
        notes="Filing PDFs are fetched per sampled document and cached; not pinned.",
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
        if fetch is None:
            raise ValueError("financebench needs a fetcher for the filing PDFs")
        rows = _jsonl(paths["financebench_open_source.jsonl"])
        docs = {r["doc_name"] for r in _jsonl(paths["financebench_document_information.jsonl"])}
        docs |= {r["doc_name"] for r in rows}
        by_id = {r["financebench_id"]: r for r in rows}
        relevant = {qid: {r["doc_name"]} for qid, r in by_id.items()}
        qids, doc_names = sample_corpus(relevant, docs, limit=limit, questions=questions, seed=seed)
        documents: dict[str, bytes] = {}
        files: list[dict] = []
        for doc in doc_names:
            remote = RemoteFile(url=pdf_url(doc), filename=safe_filename(doc) + ".pdf")
            local = fetch(remote)
            documents[remote.filename] = local.read_bytes()
            files.append(
                {"url": remote.url, "filename": remote.filename, "sha256": sha256_of(local)}
            )

        def qtype(row: dict) -> QuestionType:
            reasoning = (row.get("question_reasoning") or "").lower()
            if "numerical reasoning" in reasoning:
                return QuestionType.COMPLEX_REASONING
            return QuestionType.SIMPLE_FACTUAL

        built = [
            EvalQuestion(
                id=f"financebench-{qid}",
                question=by_id[qid]["question"],
                expected_answer=str(by_id[qid].get("answer") or ""),
                expected_sources=[
                    ExpectedSource(document=safe_filename(by_id[qid]["doc_name"]) + ".pdf")
                ],
                question_type=qtype(by_id[qid]),
                difficulty=by_id[qid].get("question_type") or "",
            )
            for qid in qids
        ]
        chosen_rel = {d for q in qids for d in relevant[q]}
        return ImportedDataset(
            documents=documents,
            questions=built,
            description=(
                "FinanceBench open sample (150 questions over SEC filings), imported by "
                "ragfabric eval import. Expected sources name the filing only."
            ),
            dropped=[
                {
                    "what": "question",
                    "count": len(relevant) - len(qids),
                    "reason": "not sampled: over the question cap or its relevant documents "
                    "did not fit in half of the document budget",
                }
            ],
            counts={
                "relevant_documents": len(chosen_rel),
                "distractors": len(doc_names) - len(chosen_rel),
                "source_documents": len(docs),
                "source_questions": len(rows),
            },
            files=files,
        )


FINANCEBENCH = FinanceBench()
