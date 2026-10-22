"""Score answers written outside RagFabric on the chunks RagFabric retrieved (ADR 0004, 0015).

``export_prompts`` runs retrieval only and writes, per question, the passages and the exact
messages the normal generator would send. Someone else's model answers them. ``import_answers``
reads those answers back and scores them through ``run_batch``, the code a normal run uses, so
retrieval metrics, citation correctness, refusal counting and the judge are the same code.

Nothing here calls a model API. An answer that is missing, duplicated, for an unknown question or
unreadable is a named, counted error on the stored run, never skipped silently and never scored
as zero. Every stored run carries who answered, where the answers came from, the hash of the
prompt file and the export date, so it cannot be read as a RagFabric API run.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy.orm import Session

from ragfabric_core.config_file import RagFabricConfig
from ragfabric_core.evaluation.dataset import EvalQuestion, QuestionSet
from ragfabric_core.evaluation.judge import Judge, build_judge
from ragfabric_core.evaluation.runner import _existing_run, run_batch
from ragfabric_core.evaluation.session import judge_model
from ragfabric_core.evaluation.strategy_target import StrategyTarget
from ragfabric_core.evaluation.target import ContextItem, TargetAnswer
from ragfabric_core.generate.cited import SYSTEM_PROMPT, build_prompt
from ragfabric_core.providers.base import Message
from ragfabric_core.providers.registry import build_llm_provider

FORMAT = "ragfabric-external-prompts-v1"
ANSWER_SOURCE = "external answers file"
EXPORT_STRATEGIES = ("traditional", "vectorless")

# The named import errors, in the order they are reported.
ERROR_KINDS = ("unreadable_line", "invalid_record", "unknown_id", "duplicate_id", "missing_answer")


class ExportError(ValueError):
    pass


class PromptFileError(ValueError):
    """The prompts file cannot be used at all (no header, bad line, duplicate question)."""


def slug(label: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")
    if not text:
        raise ValueError("--label needs at least one letter or digit")
    return text


def target_name(label: str) -> str:
    return f"external:{slug(label)}"


def prompt_messages(question: str, chunks) -> list[Message]:
    """The messages the cited generator sends for these chunks: system, then the user turn."""
    return [
        Message(role="system", content=SYSTEM_PROMPT),
        Message(role="user", content=build_prompt(question, chunks)),
    ]


# ---------------------------------------------------------------- export


def export_prompts(
    questions: QuestionSet,
    target: StrategyTarget,
    out: Path,
    *,
    header: dict,
    on_question: Callable[[str, int], None] | None = None,
) -> int:
    """Write the header line and one line per question to ``out``; return the question count.

    Retrieval runs through the target's own strategy and access context, so the passages are the
    ones a normal run would hand to the generator. No model is called for generation or judging.
    """
    if target.name not in EXPORT_STRATEGIES:
        raise ExportError(f"export-prompts supports {', '.join(EXPORT_STRATEGIES)}")
    lines = [json.dumps({"type": "header", "format": FORMAT, **header})]
    for question in questions.questions:
        result = target.retrieve_result(question.question)
        passages = [
            {
                "number": i,
                "chunk_id": c.chunk_id,
                "document": target.document_name(c.document_id, c.metadata),
                "text": c.text,
                "score": c.score,
            }
            for i, c in enumerate(result.chunks, start=1)
        ]
        messages = prompt_messages(question.question, result.chunks)
        lines.append(
            json.dumps(
                {
                    "type": "question",
                    "id": question.id,
                    "question": question.question,
                    "passages": passages,
                    "messages": [m.model_dump() for m in messages],
                    "ground_truth": {
                        "expected_answer": question.expected_answer,
                        "expected_sources": [s.model_dump() for s in question.expected_sources],
                        "question_type": str(question.question_type),
                        "difficulty": question.difficulty,
                    },
                    "retrieval": {
                        "strategy_used": str(result.strategy),
                        "retrieval_calls": result.retrieval_calls,
                        "embedding_calls": result.embedding_calls,
                    },
                }
            )
        )
        if on_question is not None:
            on_question(question.id, len(passages))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(questions.questions)


def export_header(*, corpus: str, collection: str, top_k: int, strategy: str, **extra) -> dict:
    return {
        "corpus": corpus,
        "collection": collection,
        "top_k": top_k,
        "strategy": strategy,
        "exported_at": datetime.now(UTC).strftime("%Y-%m-%d"),
        **extra,
    }


# ---------------------------------------------------------------- import


@dataclass
class ExportedQuestion:
    question: EvalQuestion
    contexts: list[ContextItem]
    messages: list[dict]
    retrieval: dict


@dataclass
class PromptFile:
    header: dict
    questions: list[ExportedQuestion]
    sha256: str


def read_prompts(path: Path) -> PromptFile:
    if not path.is_file():
        raise PromptFileError(f"prompts file not found: {path}")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    header: dict | None = None
    questions: list[ExportedQuestion] = []
    seen: set[str] = set()
    for number, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError("not a JSON object")
            if record.get("type") == "header":
                if header is not None or questions:
                    raise ValueError("header must be the single first line")
                if record.get("format") != FORMAT:
                    raise ValueError(f"unsupported format {record.get('format')!r}")
                header = record
                continue
            if header is None:
                raise ValueError("the first line must be the header")
            truth = record["ground_truth"]
            question = EvalQuestion.model_validate(
                {
                    "id": record["id"],
                    "question": record["question"],
                    "expected_answer": truth.get("expected_answer", ""),
                    "expected_sources": truth.get("expected_sources", []),
                    "question_type": truth["question_type"],
                    "difficulty": truth.get("difficulty", ""),
                }
            )
            contexts = [
                ContextItem(
                    rank=p["number"], document=p["document"], text=p["text"], score=p.get("score")
                )
                for p in record["passages"]
            ]
            messages = record["messages"]
        except Exception as exc:
            raise PromptFileError(f"{path} line {number}: {type(exc).__name__}: {exc}") from exc
        if question.id in seen:
            raise PromptFileError(f"{path} line {number}: duplicate question id {question.id!r}")
        seen.add(question.id)
        questions.append(
            ExportedQuestion(question, contexts, messages, record.get("retrieval") or {})
        )
    if header is None:
        raise PromptFileError(f"{path} has no header line")
    if not questions:
        raise PromptFileError(f"{path} has no questions")
    return PromptFile(header, questions, digest)


@dataclass
class AnswerRecord:
    answer: str
    model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass
class AnswerFile:
    answers: dict[str, AnswerRecord] = field(default_factory=dict)
    # question ids that appeared more than once: none of their answers is used
    duplicates: set[str] = field(default_factory=set)
    errors: list[dict] = field(default_factory=list)


def _token(record: dict, key: str) -> int | None:
    value = record.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{key} must be a non-negative integer or null")
    return value


def read_answers(path: Path, known_ids: set[str]) -> AnswerFile:
    """Parse an answers file; every line that cannot be used becomes a named error."""
    if not path.is_file():
        raise PromptFileError(f"answers file not found: {path}")
    out = AnswerFile()
    for number, line in enumerate(path.read_bytes().decode("utf-8", "replace").splitlines(), 1):
        if not line.strip():
            continue

        def fail(kind: str, detail: str, qid: str | None = None, _n: int = number) -> None:
            out.errors.append({"kind": kind, "line": _n, "id": qid, "detail": detail})

        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            fail("unreadable_line", f"not valid JSON: {exc.msg}")
            continue
        if not isinstance(record, dict):
            fail("unreadable_line", "not a JSON object")
            continue
        qid, text = record.get("id"), record.get("answer")
        if not isinstance(qid, str) or not qid:
            fail("invalid_record", "id must be a non-empty string")
            continue
        if not isinstance(text, str) or not text.strip():
            fail("invalid_record", "answer must be a non-empty string", qid)
            continue
        try:
            tokens = (_token(record, "input_tokens"), _token(record, "output_tokens"))
            model = record.get("model")
            if model is not None and not isinstance(model, str):
                raise ValueError("model must be a string or null")
        except ValueError as exc:
            fail("invalid_record", str(exc), qid)
            continue
        if qid not in known_ids:
            fail("unknown_id", "no such question in the prompts file", qid)
            continue
        if qid in out.duplicates or qid in out.answers:
            out.answers.pop(qid, None)
            out.duplicates.add(qid)
            fail("duplicate_id", "answered more than once; none of its answers is used", qid)
            continue
        out.answers[qid] = AnswerRecord(text, model, tokens[0], tokens[1])
    return out


class ExternalTarget:
    """Hands the runner the stored answer for each question, in prompts file order.

    The runner asks in question order, so the next id is known; the question text is checked
    against it. A question without a usable answer raises, which the runner stores as an error
    row with ``None`` scores (the same path a failing model takes).
    """

    def __init__(self, name: str, prompts: PromptFile, answers: AnswerFile) -> None:
        self.name = name
        self._order = [q.question.id for q in prompts.questions]
        self._by_id = {q.question.id: q for q in prompts.questions}
        self._answers = answers
        self._strategy = str(prompts.header.get("strategy", "")) or None

    def answer(self, question: str) -> TargetAnswer:
        qid = self._order.pop(0)
        exported = self._by_id[qid]
        if exported.question.question != question:  # pragma: no cover - runner order contract
            raise RuntimeError(f"question order mismatch at {qid}")
        if qid in self._answers.duplicates:
            raise ValueError("duplicate_id: answered more than once, not scored")
        record = self._answers.answers.get(qid)
        if record is None:
            raise ValueError("missing_answer: no answer supplied for this question")
        retrieval = exported.retrieval
        return TargetAnswer(
            answer=record.answer,
            contexts=exported.contexts,
            strategy_used=retrieval.get("strategy_used") or self._strategy,
            retrieval_calls=retrieval.get("retrieval_calls"),
            embedding_calls=retrieval.get("embedding_calls"),
            input_tokens=record.input_tokens,
            output_tokens=record.output_tokens,
            llm_model=record.model,
        )


def build_import_judge(
    cfg: RagFabricConfig, judge: str | None, judge_metrics: tuple[str, ...] | None = None
) -> Judge:
    llm = build_llm_provider(cfg.llm)
    chosen = build_judge(judge or cfg.evaluation.judge, llm, judge_model(cfg), judge_metrics)
    if chosen.kind == "llm" and chosen.model is None:
        chosen.model = llm.default_model
    return chosen


@dataclass
class ImportResult:
    run_id: int
    target: str
    error_counts: dict[str, int]
    errors: list[dict]


def import_answers(
    prompts_path: Path,
    answers_path: Path,
    judge: Judge,
    session_factory: Callable[[], Session],
    *,
    label: str,
    batch: str,
    commit: str = "",
    ragfabric_version: str = "",
    embedding_model: str | None = None,
    on_result: Callable | None = None,
) -> ImportResult:
    """Score and store the answers as run ``external:<slug of label>`` in ``batch``."""
    name = target_name(label)
    prompts = read_prompts(prompts_path)
    answers = read_answers(answers_path, {q.question.id for q in prompts.questions})
    with session_factory() as db:
        if _existing_run(db, batch, name) is not None:
            raise ValueError(
                f"batch {batch!r} already has a run named {name!r}: use another --batch or --label"
            )
    errors = list(answers.errors)
    for exported in prompts.questions:
        qid = exported.question.id
        if qid not in answers.answers and qid not in answers.duplicates:
            errors.append(
                {"kind": "missing_answer", "line": None, "id": qid, "detail": "no answer supplied"}
            )
    counts = Counter(e["kind"] for e in errors)
    error_counts = {k: counts.get(k, 0) for k in ERROR_KINDS}
    header = prompts.header
    reported = sorted({a.model for a in answers.answers.values() if a.model})
    qs = QuestionSet(
        name=str(header.get("question_set") or "external"),
        questions=[q.question for q in prompts.questions],
    )
    meta = {
        "commit": commit,
        "ragfabric_version": ragfabric_version,
        "questions_hash": header.get("questions_hash", "n/a"),
        "corpus_hash": header.get("corpus", "n/a"),
        "collection": header.get("collection", "n/a"),
        "top_k": header.get("top_k"),
        "llm_provider": "external",
        "answered_by": label,
        "answer_source": ANSWER_SOURCE,
        "prompt_file_sha256": prompts.sha256,
        "export_date": header.get("exported_at"),
        "export_commit": header.get("commit"),
        "export_strategy": header.get("strategy"),
        "reported_models": reported or None,
        "import_errors": {"counts": error_counts, "details": errors},
    }
    (run_id,) = run_batch(
        [ExternalTarget(name, prompts, answers)],
        qs,
        judge,
        session_factory,
        batch=batch,
        meta=meta,
        llm_model=f"external: {label}",
        embedding_model=embedding_model,
        on_result=on_result,
    )
    return ImportResult(run_id, name, error_counts, errors)


def provenance(run_summary: dict) -> dict | None:
    """The provenance block of a stored run, or ``None`` for a run RagFabric answered itself."""
    meta = run_summary.get("meta") or {}
    if meta.get("answer_source") is None:
        return None
    return {
        "answered_by": meta.get("answered_by"),
        "answer_source": meta.get("answer_source"),
        "prompt_file_sha256": meta.get("prompt_file_sha256"),
        "export_date": meta.get("export_date"),
        "reported_models": meta.get("reported_models"),
    }
