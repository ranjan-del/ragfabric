"""Public dataset importers (Phase 11): convert to the Phase 8 corpus and question format.

``import_dataset(name, out, ...)`` downloads (cached), samples and writes
``corpus/``, ``questions.json`` and ``manifest.json`` under ``out``. Dataset
content is never shipped or committed; each dataset's licence is recorded as
its source states it (design D5).
"""

from __future__ import annotations

from pathlib import Path

from ragfabric_core.evaluation.datasets.base import (
    DEFAULT_CACHE_DIR,
    DEFAULT_SEED,
    Fetcher,
    fetch_cached,
    sha256_of,
    write_imported,
)
from ragfabric_core.evaluation.datasets.beir import FIQA, NFCORPUS, SCIFACT

DATASETS = {d.spec.name: d for d in (SCIFACT, NFCORPUS, FIQA)}


def import_dataset(
    name: str,
    out: Path | str,
    *,
    limit: int | None = None,
    questions: int | None = None,
    seed: int = DEFAULT_SEED,
    cache_dir: Path | str | None = None,
    fetcher: Fetcher | None = None,
    verify_checksums: bool = True,
) -> dict:
    """Import one dataset into ``out``; return the manifest written there."""
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}; known: {', '.join(sorted(DATASETS))}")
    if limit is not None and limit < 2:
        raise ValueError("limit must be at least 2 documents")
    if questions is not None and questions < 1:
        raise ValueError("questions must be at least 1")
    dataset = DATASETS[name]
    cache = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    paths: dict[str, Path] = {}
    files: list[dict] = []
    for remote in dataset.spec.files:
        local = fetch_cached(name, remote, cache, fetcher, verify=verify_checksums)
        paths[remote.filename] = local
        files.append({"url": remote.url, "filename": remote.filename, "sha256": sha256_of(local)})
    data = dataset.convert(paths, limit=limit, questions=questions, seed=seed)
    return write_imported(
        Path(out), dataset.spec, data, files=files, limit=limit, questions=questions, seed=seed
    )


__all__ = ["DATASETS", "DEFAULT_CACHE_DIR", "import_dataset"]
