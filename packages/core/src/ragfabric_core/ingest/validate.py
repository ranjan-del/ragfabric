"""Upload validation: one set of rules for the API, the CLI and every connector.

An extension is a claim made by whoever named the file. Before a file reaches a
parser, this module checks that the claim is plausible and that the file is
safe to open:

| Check | Status |
|---|---|
| Extension is a supported format and in ``limits.allowed_types`` | 400 |
| Non-empty and at most ``limits.max_upload_mb`` | 400 / 413 |
| PDF has a ``%PDF-`` header in its first kilobyte | 415 |
| DOCX and PPTX are zip files with their main part present | 415 |
| TXT, MD and CSV have no NUL byte and no binary signature | 415 |
| A zip has at most ``MAX_ZIP_ENTRIES`` entries, at most ``limits.max_uncompressed_mb`` in total, and no large entry compressed more than ``MAX_RATIO`` to 1 | 413 |

The zip checks read only the central directory, so a zip bomb is refused
without inflating a byte of it. The declared sizes are trustworthy for this
purpose because Python's ``zipfile``, which the DOCX and PPTX parsers use, never
returns more than an entry's declared size.

Text formats are not required to be UTF-8: the parser falls back to Latin-1,
and spreadsheet exports in Windows code pages are common. A NUL byte, which no
such text contains, is the signal that a file is binary.

``UploadRejected.status`` is an HTTP status because the API is the main caller;
the CLI and the connectors report ``reason`` and move on.
"""

from __future__ import annotations

import io
import zipfile

from ragfabric_core.config_file import LimitsConfig
from ragfabric_core.ingest.parser import SUPPORTED_FORMATS

MIB = 1024 * 1024
MAX_ZIP_ENTRIES = 10_000
MAX_RATIO = 100
# Entries smaller than this are not ratio-checked: small XML parts compress
# well and are harmless at any ratio.
RATIO_CHECK_MIN_BYTES = MIB
PDF_HEADER_WINDOW = 1024
TEXT_SNIFF_BYTES = 8192

_ZIP_MAIN_PART = {"docx": "word/document.xml", "pptx": "ppt/presentation.xml"}
_BINARY_SIGNATURES = {
    b"%PDF-": "a PDF",
    b"PK\x03\x04": "a zip archive (DOCX, PPTX or similar)",
    b"\x89PNG": "a PNG image",
    b"\xff\xd8\xff": "a JPEG image",
    b"GIF8": "a GIF image",
    b"\x7fELF": "an executable",
}


class UploadRejected(ValueError):
    """A file this deployment will not ingest, with the HTTP status and a reason."""

    def __init__(self, status: int, reason: str) -> None:
        super().__init__(reason)
        self.status = status
        self.reason = reason


def extension(filename: str) -> str:
    return filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


def check_type(filename: str, limits: LimitsConfig) -> str:
    """The extension checks alone, for callers that have not read the bytes yet."""
    ext = extension(filename)
    if ext not in SUPPORTED_FORMATS:
        raise UploadRejected(
            400,
            f"Unsupported format '{ext or filename}'. Supported: {', '.join(SUPPORTED_FORMATS)}.",
        )
    if ext not in limits.allowed_types:
        raise UploadRejected(
            400,
            f"File type '.{ext}' is not allowed by this deployment's configuration. "
            f"Allowed: {', '.join(limits.allowed_types)}.",
        )
    return ext


def check_size(size_bytes: int, limits: LimitsConfig) -> None:
    if size_bytes > limits.max_upload_mb * MIB:
        raise UploadRejected(
            413,
            f"File is {size_bytes / MIB:.1f} MB, over the configured limit of "
            f"{limits.max_upload_mb} MB.",
        )


def _check_pdf(data: bytes) -> None:
    if b"%PDF-" not in data[:PDF_HEADER_WINDOW]:
        raise UploadRejected(415, "File is named .pdf but is not a PDF (no %PDF- header).")


def _check_zip(ext: str, data: bytes, limits: LimitsConfig) -> None:
    kind = ext.upper()
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise UploadRejected(415, f"File is named .{ext} but is not a valid {kind} file.") from exc
    with archive:
        entries = archive.infolist()
        if len(entries) > MAX_ZIP_ENTRIES:
            raise UploadRejected(
                413, f"{kind} file has {len(entries)} entries, over the limit of {MAX_ZIP_ENTRIES}."
            )
        total = sum(entry.file_size for entry in entries)
        if total > limits.max_uncompressed_mb * MIB:
            raise UploadRejected(
                413,
                f"{kind} file would expand to {total / MIB:.0f} MB uncompressed, over the "
                f"limit of {limits.max_uncompressed_mb} MB.",
            )
        for entry in entries:
            if entry.file_size < RATIO_CHECK_MIN_BYTES:
                continue
            ratio = entry.file_size / max(entry.compress_size, 1)
            if ratio > MAX_RATIO:
                raise UploadRejected(
                    413,
                    f"{kind} file entry '{entry.filename}' has a compression ratio of "
                    f"{ratio:.0f} to 1, over the limit of {MAX_RATIO} to 1.",
                )
        names = {entry.filename for entry in entries}
        if _ZIP_MAIN_PART[ext] not in names:
            raise UploadRejected(
                415,
                f"File is named .{ext} but is not a {kind} file ({_ZIP_MAIN_PART[ext]} missing).",
            )


def _check_text(ext: str, data: bytes) -> None:
    head = data[:TEXT_SNIFF_BYTES]
    for signature, what in _BINARY_SIGNATURES.items():
        if head.startswith(signature):
            raise UploadRejected(415, f"File is named .{ext} but looks like {what}.")
    if b"\x00" in head:
        raise UploadRejected(
            415,
            f"File is named .{ext} but contains binary data (a NUL byte). "
            "Save it as UTF-8 text and upload it again.",
        )


def validate_upload(filename: str, data: bytes, limits: LimitsConfig) -> str:
    """Check ``data`` against every rule above; return the extension or raise."""
    ext = check_type(filename, limits)
    if not data:
        raise UploadRejected(400, "Uploaded file is empty.")
    check_size(len(data), limits)
    if ext == "pdf":
        _check_pdf(data)
    elif ext in _ZIP_MAIN_PART:
        _check_zip(ext, data, limits)
    else:
        _check_text(ext, data)
    return ext
