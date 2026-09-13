"""Retained originals on local disk.

Citations can only link back to a source if the source still exists. Files are
kept under ``<uploads_dir>/<document_id>/<sanitised name>`` so a document's
files are removed together and names from users can never escape the root.
Object storage is a later connector behind the same three methods.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from ragfabric_core.runtime import get_config

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_filename(name: str) -> str:
    cleaned = _UNSAFE.sub("_", name).strip("._") or "file"
    return cleaned


class FileStorage:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def save(self, document_id: int, filename: str, data: bytes) -> str:
        target_dir = self.root / str(document_id)
        target_dir.mkdir(parents=True, exist_ok=True)
        relative = f"{document_id}/{safe_filename(filename)}"
        (self.root / relative).write_bytes(data)
        return relative

    def path_for(self, relative: str) -> Path:
        candidate = (self.root / relative).resolve()
        if not candidate.is_relative_to(self.root.resolve()):
            raise ValueError("storage path escapes the uploads root")
        return candidate

    def delete(self, document_id: int) -> None:
        shutil.rmtree(self.root / str(document_id), ignore_errors=True)


_storage: FileStorage | None = None


def get_storage() -> FileStorage:
    global _storage
    root = Path(get_config().ingestion.uploads_dir)
    if _storage is None or _storage.root != root:
        _storage = FileStorage(root)
    return _storage
