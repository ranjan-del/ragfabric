"""Google Drive connector: every supported file in one Drive folder (design D22 to D24).

Built on the official ``google-api-python-client`` with a service account key
and the read-only scope. Install it with ``pip install 'ragfabric[drive]'``.
The folder is shared with the service account's address, so the operator
decides exactly what it can read, and a leaked key cannot change anything.
Setup, in full, is in docs/connectors.md.

What is listed, and how:

- Files whose parent is the folder (and, when ``recursive``, each subfolder,
  visited once however many times it is reached), not in the trash, on My
  Drive or a shared drive.
- Google Docs, Slides and Sheets are exported to DOCX, PPTX and CSV, which the
  existing parsers read. Drive exports only a Sheet's first tab, and refuses
  exports over 10 MB.
- Other files are downloaded as they are when their name or MIME type is a
  supported format. Anything else (shortcuts, forms, drawings, images,
  archives) is ignored, and named in ``ignored`` so the sync can count it.
- A binary file's version is its ``md5Checksum``; a Google native file has
  none, so its Drive ``version`` number stands in. Either way the sync engine
  fetches only what changed (D18).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ragfabric_core.connectors.base import SourceDocument
from ragfabric_core.ingest.parser import SUPPORTED_FORMATS

if TYPE_CHECKING:
    from ragfabric_core.config_file import GoogleDriveConnectorConfig

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
INSTALL_HINT = "pip install 'ragfabric[drive]'"

FOLDER_MIME = "application/vnd.google-apps.folder"
DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
PPTX_MIME = "application/vnd.openxmlformats-officedocument.presentationml.presentation"

# Google native type -> (extension added to the name, export MIME type).
EXPORTS: dict[str, tuple[str, str]] = {
    "application/vnd.google-apps.document": ("docx", DOCX_MIME),
    "application/vnd.google-apps.presentation": ("pptx", PPTX_MIME),
    "application/vnd.google-apps.spreadsheet": ("csv", "text/csv"),
}

# Downloadable MIME type -> the extension a file of that type gets when its
# name has no supported one.
BINARY_TYPES: dict[str, str] = {
    "application/pdf": "pdf",
    DOCX_MIME: "docx",
    PPTX_MIME: "pptx",
    "text/plain": "txt",
    "text/markdown": "md",
    "text/csv": "csv",
}

LIST_FIELDS = "nextPageToken, files(id, name, mimeType, size, modifiedTime, md5Checksum, version)"
NUM_RETRIES = 3


class DriveSetupError(RuntimeError):
    """The connector cannot start: the extra is missing or the key is unusable."""


def build_drive_service(credentials_file: str) -> Any:
    """A Drive v3 client authorised by a service account key, read-only.

    Raises ``DriveSetupError`` with a message an operator can act on; the key's
    content never appears in it.
    """
    try:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build
    except ImportError as exc:
        raise DriveSetupError(
            f"the Google Drive connector needs the drive extra: {INSTALL_HINT}"
        ) from exc

    path = Path(credentials_file)
    if not path.is_file():
        raise DriveSetupError(
            f"Google Drive credentials file {path} does not exist; point credentials_file "
            "at a service account key (see docs/connectors.md)"
        )
    problem = "it is not valid JSON"
    try:
        info = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(info, dict) or info.get("type") != "service_account":
            problem = 'its "type" is not "service_account"'
            raise ValueError(problem)
        problem = "it is missing fields a service account key has"
        credentials = service_account.Credentials.from_service_account_info(info, scopes=SCOPES)
    except (OSError, ValueError) as exc:
        if isinstance(exc, OSError):
            problem = "it could not be read"
        raise DriveSetupError(
            f"Google Drive credentials file {path} is not a readable service account key: {problem}"
        ) from None
    return build(
        "drive", "v3", credentials=credentials, cache_discovery=False, static_discovery=True
    )


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


class GoogleDriveConnector:
    name = "google_drive"

    def __init__(self, service: Any, folder_id: str, *, recursive: bool = True) -> None:
        self.service = service
        self.folder_id = folder_id
        self.recursive = recursive
        # "<id> (<name>): <mime type>" for every file the last listing passed over.
        self.ignored: list[str] = []
        self._mime: dict[str, str] = {}

    @classmethod
    def from_config(cls, cfg: GoogleDriveConnectorConfig) -> GoogleDriveConnector:
        return cls(
            build_drive_service(cfg.credentials_file), cfg.folder_id, recursive=cfg.recursive
        )

    def _children(self, folder_id: str) -> Iterator[dict[str, Any]]:
        files = self.service.files()
        token: str | None = None
        while True:
            kwargs: dict[str, Any] = {
                "q": f"'{_escape(folder_id)}' in parents and trashed = false",
                "fields": LIST_FIELDS,
                "pageSize": 1000,
                "supportsAllDrives": True,
                "includeItemsFromAllDrives": True,
            }
            if token:
                kwargs["pageToken"] = token
            page = files.list(**kwargs).execute(num_retries=NUM_RETRIES)
            yield from page.get("files", [])
            token = page.get("nextPageToken")
            if not token:
                return

    def _source(self, item: dict[str, Any]) -> SourceDocument | None:
        mime = item.get("mimeType", "")
        name = item.get("name", "") or item["id"]
        modified_at = _parse_time(item.get("modifiedTime"))
        if mime in EXPORTS:
            extension, export_mime = EXPORTS[mime]
            return SourceDocument(
                source_id=item["id"],
                name=f"{name}.{extension}",
                content_type=export_mime,
                modified_at=modified_at,
                version=f"v{item['version']}" if item.get("version") else "",
            )
        if mime.startswith("application/vnd.google-apps."):
            return None
        extension = Path(name).suffix.lower().lstrip(".")
        if extension not in SUPPORTED_FORMATS:
            if mime not in BINARY_TYPES:
                return None
            name = f"{name}.{BINARY_TYPES[mime]}"
        size = item.get("size")
        return SourceDocument(
            source_id=item["id"],
            name=name,
            content_type=mime,
            size_bytes=int(size) if size is not None else None,
            modified_at=modified_at,
            version=item.get("md5Checksum") or "",
        )

    def _check_root(self) -> None:
        """Look the folder up before listing it.

        Listing the children of an id that does not exist, or that is not
        shared with the service account, returns 200 and no files, which is
        indistinguishable from an empty folder. A first sync would report
        nothing and succeed; this makes it say what is wrong.
        """
        from googleapiclient.errors import HttpError

        try:
            meta = (
                self.service.files()
                .get(fileId=self.folder_id, fields="id, name, mimeType", supportsAllDrives=True)
                .execute(num_retries=NUM_RETRIES)
            )
        except HttpError as exc:
            status = getattr(exc.resp, "status", None)
            if str(status) in ("403", "404"):
                raise DriveSetupError(
                    f"Google Drive folder {self.folder_id} was not found or not shared with the "
                    f"service account (HTTP {status}); check folder_id and share the folder with "
                    "the service account's address (docs/connectors.md)"
                ) from None
            raise
        if meta.get("mimeType") != FOLDER_MIME:
            raise DriveSetupError(
                f"Google Drive id {self.folder_id} is not a folder ({meta.get('mimeType')}); "
                "folder_id must name a folder"
            )

    def list_documents(self) -> Iterator[SourceDocument]:
        self.ignored = []
        self._check_root()
        pending = [self.folder_id]
        seen = {self.folder_id}
        while pending:
            folder = pending.pop(0)
            for item in self._children(folder):
                mime = item.get("mimeType", "")
                if mime == FOLDER_MIME:
                    if self.recursive and item["id"] not in seen:
                        seen.add(item["id"])
                        pending.append(item["id"])
                    continue
                source = self._source(item)
                if source is None:
                    self.ignored.append(f"{item['id']} ({item.get('name', '')}): {mime}")
                    continue
                self._mime[item["id"]] = mime
                yield source

    def fetch(self, source_id: str) -> bytes:
        files = self.service.files()
        mime = self._mime.get(source_id)
        if mime is None:
            meta = files.get(
                fileId=source_id, fields="id, name, mimeType", supportsAllDrives=True
            ).execute(num_retries=NUM_RETRIES)
            mime = meta.get("mimeType", "")
            self._mime[source_id] = mime
        if mime in EXPORTS:
            request = files.export_media(fileId=source_id, mimeType=EXPORTS[mime][1])
        else:
            request = files.get_media(fileId=source_id, supportsAllDrives=True)
        return request.execute(num_retries=NUM_RETRIES)
