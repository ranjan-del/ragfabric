"""Importer core: download with a checksum and a cache, sample, write a Phase 8 dataset.

Standard library only (design D2): ``pip install ragfabric`` users must be able
to run ``ragfabric eval import`` without ``datasets`` or ``pyarrow``.

Downloads go to a cache (D3), never into the repository. A file whose
publisher checksum is pinned is verified against it; an unpinned file is
verified against the checksum recorded on its first download, so a truncated
or edited cache entry is fetched again rather than silently converted.

Sampling (``limit``, ``questions``, ``seed``) is deterministic: the same
arguments always give byte identical files.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import shutil
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

from ragfabric_core.evaluation.dataset import EvalQuestion, QuestionSet, load_questions

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "ragfabric" / "datasets"
DEFAULT_SEED = 11

Fetcher = Callable[[str, Path], None]


class ChecksumError(ValueError):
    """A downloaded file does not match the checksum pinned for it."""


@dataclass(frozen=True)
class RemoteFile:
    url: str
    filename: str
    sha256: str | None = None


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    title: str
    source: str
    licence: str
    files: tuple[RemoteFile, ...]
    notes: str = ""


@dataclass
class ImportedDataset:
    """What a converter returns: documents to write, questions, and what it dropped."""

    documents: dict[str, bytes]
    questions: list[EvalQuestion]
    description: str
    dropped: list[dict] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)


def urllib_fetcher(url: str, dest: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "ragfabric-eval-import"})
    with urllib.request.urlopen(request, timeout=600) as response, dest.open("wb") as fh:
        shutil.copyfileobj(response, fh)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_cached(
    dataset: str,
    remote: RemoteFile,
    cache_dir: Path,
    fetcher: Fetcher | None = None,
    *,
    verify: bool = True,
) -> Path:
    """Return a local copy of ``remote``, downloading only when the cache is missing or wrong."""
    folder = Path(cache_dir) / dataset
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / remote.filename
    sidecar = folder / (remote.filename + ".sha256")
    pinned = remote.sha256 if verify else None
    expected = pinned or (sidecar.read_text().strip() if sidecar.is_file() else None)
    if path.is_file() and expected is not None and sha256_of(path) == expected:
        return path
    partial = folder / (remote.filename + ".part")
    (fetcher or urllib_fetcher)(remote.url, partial)
    actual = sha256_of(partial)
    if pinned is not None and actual != pinned:
        partial.unlink(missing_ok=True)
        raise ChecksumError(
            f"{remote.filename} from {remote.url} has sha256 {actual}, expected {pinned}"
        )
    partial.replace(path)
    sidecar.write_text(actual + "\n")
    return path


_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")


def safe_filename(raw: str) -> str:
    """A document id as a file name: unsafe characters become ``_``, no leading dot."""
    name = _UNSAFE.sub("_", raw.strip()) or "_"
    return "_" + name if name.startswith(".") else name


def sample_corpus(
    relevant: dict[str, set[str]],
    documents: Iterable[str],
    *,
    limit: int | None,
    questions: int | None,
    seed: int,
) -> tuple[list[str], list[str]]:
    """Pick question ids and document ids; both lists are sorted.

    Questions are shuffled with ``random.Random(seed)`` over their sorted ids.
    A question is taken when the union of chosen relevant documents still fits
    in half of ``limit``; the rest of the budget is filled with distractors
    drawn from the same generator. ``limit`` None means every document.
    """
    rng = random.Random(seed)
    order = sorted(relevant)
    rng.shuffle(order)
    all_docs = sorted(set(documents))
    budget = None if limit is None else limit // 2
    chosen_q: list[str] = []
    chosen_docs: set[str] = set()
    for qid in order:
        if questions is not None and len(chosen_q) >= questions:
            break
        union = chosen_docs | relevant[qid]
        if budget is not None and len(union) > budget:
            continue
        chosen_q.append(qid)
        chosen_docs = union
    if limit is None:
        return sorted(chosen_q), all_docs
    pool = [d for d in all_docs if d not in chosen_docs]
    room = max(0, min(limit - len(chosen_docs), len(pool)))
    return sorted(chosen_q), sorted(chosen_docs | set(rng.sample(pool, room)))


def write_imported(
    out: Path,
    spec: DatasetSpec,
    data: ImportedDataset,
    *,
    files: list[dict],
    limit: int | None,
    questions: int | None,
    seed: int,
) -> dict:
    """Write ``corpus/``, ``questions.json`` and ``manifest.json``; return the manifest."""
    out = Path(out)
    question_set = QuestionSet(
        name=f"{spec.name}" + (f"-{limit}" if limit is not None else "-full"),
        description=data.description,
        questions=data.questions,
    )
    corpus = out / "corpus"
    if corpus.exists():
        shutil.rmtree(corpus)
    corpus.mkdir(parents=True)
    for filename, content in sorted(data.documents.items()):
        (corpus / filename).write_bytes(content)
    questions_path = out / "questions.json"
    questions_path.write_text(
        json.dumps(question_set.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    load_questions(questions_path)  # fail here, not at run time, if the schema is broken
    manifest = {
        "dataset": spec.name,
        "title": spec.title,
        "source": spec.source,
        "licence": spec.licence,
        "notes": spec.notes,
        "files": files,
        "sampling": {"limit": limit, "questions": questions, "seed": seed},
        "counts": {"documents": len(data.documents), "questions": len(data.questions)}
        | data.counts,
        "dropped": [d for d in data.dropped if d.get("count")],
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest
