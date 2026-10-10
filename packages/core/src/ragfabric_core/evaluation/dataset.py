"""Question sets: the shipped one and anyone's own, in one schema.

``expected_sources`` names a document and, optionally, a short evidence
phrase from it (design decision D3). Chunk ids would change with every
re-ingest and every ``chunk_size``; a document name plus a phrase survives
both and still tells the right passage from the wrong one in the same file.

The shipped corpus and questions live inside this package so a ``pip
install`` user runs exactly the set ``make eval`` runs (D2).
"""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from importlib.resources import files
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class QuestionType(StrEnum):
    SIMPLE_FACTUAL = "simple_factual"
    MULTI_DOCUMENT = "multi_document"
    MULTI_HOP = "multi_hop"
    COMPARISON = "comparison"
    RELATIONSHIP = "relationship"
    EXACT_MATCH = "exact_match"
    AMBIGUOUS = "ambiguous"
    COMPLEX_REASONING = "complex_reasoning"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ExpectedSource(_Strict):
    document: str = Field(min_length=1)
    evidence: str | None = None


class EvalQuestion(_Strict):
    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    expected_answer: str = ""
    expected_sources: list[ExpectedSource] = Field(default_factory=list)
    question_type: QuestionType
    difficulty: str = ""


class QuestionSet(_Strict):
    version: int = 1
    name: str = Field(min_length=1)
    description: str = ""
    questions: list[EvalQuestion] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_ids(self) -> QuestionSet:
        seen: set[str] = set()
        for q in self.questions:
            if q.id in seen:
                raise ValueError(f"duplicate question id {q.id!r}")
            seen.add(q.id)
        return self


def _data_dir():
    return files("ragfabric_core") / "evaluation" / "data"


def load_questions(path: Path | str) -> QuestionSet:
    """Load and validate a question file, naming every bad field on failure."""
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"question file not found: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc
    try:
        return QuestionSet.model_validate(raw)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc']) or '(file)'}: {err['msg']}"
            for err in exc.errors()
        )
        raise ValueError(f"{path} is not a valid question set: {problems}") from None


def shipped_questions_path() -> Path:
    return Path(str(_data_dir() / "questions.json"))


def shipped_questions() -> QuestionSet:
    return load_questions(shipped_questions_path())


def shipped_corpus_paths() -> list[Path]:
    corpus = Path(str(_data_dir() / "corpus"))
    return sorted(p for p in corpus.iterdir() if p.suffix == ".md")


def content_hash(paths: list[Path]) -> str:
    """A short, stable fingerprint of files' contents, recorded with every run."""
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda p: p.name):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:12]
