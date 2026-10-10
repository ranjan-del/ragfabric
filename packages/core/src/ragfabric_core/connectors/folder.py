"""Watched folder connector: every supported file under one directory.

Polled, not event driven (design D20): filesystem events do not cross Docker
bind mounts from a macOS or Windows host, nor network shares, which is where a
dropped-file inbox usually lives, and scanning a folder is cheap.

What is listed, and what is not:

- Supported formats only (``SUPPORTED_FORMATS``); anything else is ignored
  rather than reported, because an inbox legitimately holds other files.
- Hidden files and folders, Office lock files (``~$``) and download or editor
  temporaries (``.tmp``, ``.part``, ``.crdownload``) are ignored: they are
  half-files by definition.
- A symlink is followed only if it resolves inside the folder. One that points
  elsewhere (``/etc``, a home directory) is ignored, so the inbox cannot be
  used to ingest files the operator never put there.
- A file modified within ``settle_seconds`` is listed as not ready, so a file
  still being copied is neither ingested half-written nor, if already known,
  dropped from the index while it is re-saved.
"""

from __future__ import annotations

import mimetypes
import time
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path

from ragfabric_core.connectors.base import SourceDocument
from ragfabric_core.ingest.parser import SUPPORTED_FORMATS

TEMP_SUFFIXES = (".tmp", ".part", ".crdownload", ".partial", ".swp")


def _ignored(relative: Path) -> bool:
    if any(part.startswith(".") for part in relative.parts):
        return True
    name = relative.name
    return name.startswith("~$") or name.lower().endswith(TEMP_SUFFIXES)


class FolderConnector:
    name = "folder"

    def __init__(
        self,
        root: str | Path,
        *,
        recursive: bool = True,
        settle_seconds: float = 5,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.root = Path(root)
        self.recursive = recursive
        self.settle_seconds = settle_seconds
        self._clock = clock

    def _real_root(self) -> Path:
        if not self.root.is_dir():
            raise FileNotFoundError(
                f"connector folder {self.root} does not exist or is not a directory"
            )
        return self.root.resolve()

    def _inside(self, path: Path, real_root: Path) -> bool:
        try:
            return path.resolve().is_relative_to(real_root)
        except OSError:
            return False

    def list_documents(self) -> Iterator[SourceDocument]:
        real_root = self._real_root()
        candidates = self.root.rglob("*") if self.recursive else self.root.glob("*")
        now = self._clock()
        for path in sorted(candidates):
            relative = path.relative_to(self.root)
            if _ignored(relative):
                continue
            if path.suffix.lower().lstrip(".") not in SUPPORTED_FORMATS:
                continue
            # Every component, not just the file: a link to a directory outside
            # the folder makes every file under it resolve outside too.
            if not self._inside(path, real_root) or not path.is_file():
                continue
            stat = path.stat()
            yield SourceDocument(
                source_id=relative.as_posix(),
                name=path.name,
                content_type=mimetypes.guess_type(path.name)[0] or "",
                size_bytes=stat.st_size,
                modified_at=datetime.fromtimestamp(stat.st_mtime, UTC),
                ready=now - stat.st_mtime >= self.settle_seconds,
            )

    def fetch(self, source_id: str) -> bytes:
        real_root = self._real_root()
        path = self.root / source_id
        if not self._inside(path, real_root):
            raise PermissionError(f"{source_id} is outside the connector folder")
        return path.read_bytes()
