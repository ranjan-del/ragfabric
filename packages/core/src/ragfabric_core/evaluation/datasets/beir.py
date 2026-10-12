"""SciFact, NFCorpus and FiQA-2018 from the BEIR zips (UKP Darmstadt).

BEIR gives relevance judgements, not answers, so ``expected_answer`` is empty
and these sets measure retrieval only (design D6). Every judgement with a
score above 0 becomes an ``expected_source`` naming the document only.
The ``test`` split is used: it is the one BEIR reports results on.
"""

from __future__ import annotations

import csv
import io
import json
import zipfile
from pathlib import Path

from ragfabric_core.evaluation.dataset import EvalQuestion, ExpectedSource, QuestionType
from ragfabric_core.evaluation.datasets.base import (
    DatasetSpec,
    ImportedDataset,
    RemoteFile,
    safe_filename,
    sample_corpus,
)

BEIR_BASE = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"


def _spec(name: str, title: str, sha256: str, licence: str, notes: str = "") -> DatasetSpec:
    return DatasetSpec(
        name=name,
        title=title,
        source=f"BEIR benchmark zip, {BEIR_BASE}/{name}.zip",
        licence=licence,
        files=(RemoteFile(url=f"{BEIR_BASE}/{name}.zip", filename=f"{name}.zip", sha256=sha256),),
        notes=notes,
    )


class BeirDataset:
    def __init__(self, spec: DatasetSpec, split: str = "test") -> None:
        self.spec = spec
        self.split = split

    def _member(self, zf: zipfile.ZipFile, suffix: str) -> str:
        for name in zf.namelist():
            if name.endswith(suffix):
                return name
        raise ValueError(f"{self.spec.name}: {suffix} not found in the zip")

    def convert(
        self, paths: dict[str, Path], *, limit: int | None, questions: int | None, seed: int
    ) -> ImportedDataset:
        name = self.spec.name
        with zipfile.ZipFile(paths[self.spec.files[0].filename]) as zf:
            docs: dict[str, tuple[str, str]] = {}
            with zf.open(self._member(zf, "/corpus.jsonl")) as fh:
                for line in io.TextIOWrapper(fh, encoding="utf-8"):
                    if line.strip():
                        row = json.loads(line)
                        docs[str(row["_id"])] = (row.get("title") or "", row.get("text") or "")
            queries: dict[str, str] = {}
            with zf.open(self._member(zf, "/queries.jsonl")) as fh:
                for line in io.TextIOWrapper(fh, encoding="utf-8"):
                    if line.strip():
                        row = json.loads(line)
                        queries[str(row["_id"])] = row["text"]
            with zf.open(self._member(zf, f"/qrels/{self.split}.tsv")) as fh:
                rows = list(csv.reader(io.TextIOWrapper(fh, encoding="utf-8"), delimiter="\t"))

        relevant: dict[str, set[str]] = {}
        zero = missing_doc = missing_query = 0
        for row in rows[1:]:
            if len(row) < 3:
                continue
            qid, did, score = row[0], row[1], int(row[2])
            if score <= 0:
                zero += 1
                continue
            if qid not in queries:
                missing_query += 1
                continue
            if did not in docs:
                missing_doc += 1
                continue
            relevant.setdefault(qid, set()).add(did)

        qids, doc_ids = sample_corpus(relevant, docs, limit=limit, questions=questions, seed=seed)
        chosen_rel = {d for q in qids for d in relevant[q]}
        documents = {
            safe_filename(d) + ".txt": (
                (docs[d][0] + "\n\n" + docs[d][1]) if docs[d][0] else docs[d][1]
            ).encode("utf-8")
            for d in doc_ids
        }
        built = [
            EvalQuestion(
                id=f"{name}-{q}",
                question=queries[q],
                expected_sources=[
                    ExpectedSource(document=safe_filename(d) + ".txt") for d in sorted(relevant[q])
                ],
                question_type=QuestionType.SIMPLE_FACTUAL,
                difficulty="",
            )
            for q in qids
        ]
        judged = len(relevant)
        return ImportedDataset(
            documents=documents,
            questions=built,
            description=(
                f"{self.spec.title}, BEIR {self.split} split, imported by ragfabric eval import. "
                "Relevance judgements only: no reference answers, so retrieval metrics only."
            ),
            dropped=[
                {"what": "judgement", "count": zero, "reason": "relevance score is 0"},
                {
                    "what": "judgement",
                    "count": missing_doc,
                    "reason": "judgement names a document missing from the corpus",
                },
                {
                    "what": "judgement",
                    "count": missing_query,
                    "reason": "judgement names a query missing from queries.jsonl",
                },
                {
                    "what": "question",
                    "count": judged - len(qids),
                    "reason": "not sampled: over the question cap or its relevant documents "
                    "did not fit in half of the document budget",
                },
            ],
            counts={
                "relevant_documents": len(chosen_rel),
                "distractors": len(doc_ids) - len(chosen_rel),
                "source_documents": len(docs),
                "source_questions_judged": judged,
            },
        )


SCIFACT = BeirDataset(
    _spec(
        "scifact",
        "SciFact (BEIR)",
        "536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165",
        "CC BY 4.0 per allenai/scifact LICENSE.md (abstracts ODC-By 1.0, from S2ORC); "
        "the BeIR/scifact Hugging Face card states CC BY-SA 4.0",
    )
)
NFCORPUS = BeirDataset(
    _spec(
        "nfcorpus",
        "NFCorpus (BEIR)",
        "efe5be03f8c5b86a5870102d0599d227c8c6e2484328e68c6522560385671b0b",
        "Original site (Heidelberg University, StatNLP): free to use for academic purposes; "
        "other uses fall under the NutritionFacts.org terms. Not confirmed for other use",
    )
)
FIQA = BeirDataset(
    _spec(
        "fiqa",
        "FiQA-2018 (BEIR)",
        "32c7df99ed21252fdfb2cf3f5673502a8d245ee0c44c4a133570d92ce2b3ad02",
        "FiQA 2018 challenge site: data available for non-commercial use only; no named licence",
    )
)
