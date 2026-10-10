"""Upload validation shared by the API, the CLI and the connectors (Phase 10, Task 3)."""

from __future__ import annotations

import io
import zipfile

import pytest

from ragfabric_core.config_file import LimitsConfig
from ragfabric_core.ingest.validate import (
    MAX_ZIP_ENTRIES,
    UploadRejected,
    validate_upload,
)
from ragfabric_core.testing.fixtures import make_csv, make_docx, make_pdf, make_pptx, make_txt

LIMITS = LimitsConfig()


def _zip(members: dict[str, bytes], compression=zipfile.ZIP_DEFLATED) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=compression) as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    return buffer.getvalue()


def _rejected(filename: str, data: bytes, limits: LimitsConfig = LIMITS) -> UploadRejected:
    with pytest.raises(UploadRejected) as info:
        validate_upload(filename, data, limits)
    return info.value


@pytest.mark.parametrize(
    ("filename", "build", "ext"),
    [
        ("a.pdf", lambda: make_pdf(["hello"]), "pdf"),
        ("A.DOCX", lambda: make_docx(["hello"]), "docx"),
        ("a.pptx", lambda: make_pptx([("t", "b")]), "pptx"),
        ("a.txt", lambda: make_txt("hello"), "txt"),
        ("a.md", lambda: b"# Title\n\nbody", "md"),
        ("a.csv", lambda: make_csv(["a", "b"], [["1", "2"]]), "csv"),
        ("latin.txt", lambda: "caf\xe9 cr\xe8me".encode("latin-1"), "txt"),
    ],
)
def test_genuine_files_pass_and_return_their_extension(filename, build, ext):
    assert validate_upload(filename, build(), LIMITS) == ext


def test_an_unknown_extension_is_400():
    err = _rejected("a.exe", b"MZ....")
    assert err.status == 400 and "Unsupported format" in err.reason


def test_a_supported_type_outside_the_allow_list_is_400():
    err = _rejected("a.pdf", make_pdf(["x"]), LimitsConfig(allowed_types=["txt"]))
    assert err.status == 400 and "not allowed" in err.reason


def test_no_extension_is_400():
    assert _rejected("README", b"text").status == 400


def test_empty_is_400():
    assert _rejected("a.txt", b"").status == 400


def test_over_the_size_limit_is_413():
    err = _rejected("a.txt", b"x" * (1024 * 1024 + 1), LimitsConfig(max_upload_mb=1))
    assert err.status == 413 and "1 MB" in err.reason


def test_a_pdf_without_the_pdf_signature_is_415():
    err = _rejected("a.pdf", b"this is not a pdf at all")
    assert err.status == 415 and "PDF" in err.reason


def test_a_pdf_signature_after_leading_whitespace_is_accepted():
    """Some generators emit a few bytes before %PDF-; readers accept a header in
    the first kilobyte, and so does the validator."""
    assert validate_upload("a.pdf", b"\n\n" + make_pdf(["x"]), LIMITS) == "pdf"


def test_a_docx_that_is_not_a_zip_is_415():
    assert _rejected("a.docx", b"plain text pretending").status == 415


def test_a_zip_without_the_word_document_part_is_415():
    err = _rejected("a.docx", _zip({"hello.txt": b"hi"}))
    assert err.status == 415 and "DOCX" in err.reason


def test_a_docx_renamed_to_pptx_is_415():
    err = _rejected("a.pptx", make_docx(["hello"]))
    assert err.status == 415 and "PPTX" in err.reason


@pytest.mark.parametrize("filename", ["a.txt", "a.md", "a.csv"])
def test_text_with_a_nul_byte_is_415(filename):
    err = _rejected(filename, b"looks like text\x00\x01\x02")
    assert err.status == 415 and "binary" in err.reason


def test_a_pdf_renamed_to_txt_is_415():
    err = _rejected("a.txt", make_pdf(["hello"]))
    assert err.status == 415


def test_a_zip_over_the_uncompressed_limit_is_413_without_decompressing():
    data = _zip({"word/document.xml": b"<w/>", "big.bin": b"\x00" * (3 * 1024 * 1024)})
    err = _rejected("a.docx", data, LimitsConfig(max_uncompressed_mb=2))
    assert err.status == 413 and "uncompressed" in err.reason


def test_a_zip_entry_with_an_extreme_compression_ratio_is_413():
    data = _zip({"word/document.xml": b"<w/>", "bomb.xml": b"A" * (20 * 1024 * 1024)})
    err = _rejected("a.docx", data)
    assert err.status == 413 and "ratio" in err.reason


def test_a_zip_with_too_many_entries_is_413(monkeypatch):
    import ragfabric_core.ingest.validate as validate

    monkeypatch.setattr(validate, "MAX_ZIP_ENTRIES", 3)
    data = _zip({"word/document.xml": b"<w/>", **{f"f{i}": b"x" for i in range(5)}})
    err = _rejected("a.docx", data)
    assert err.status == 413 and "entries" in err.reason
    assert MAX_ZIP_ENTRIES == 10_000


def test_the_status_and_reason_are_readable_from_the_exception():
    err = _rejected("a.exe", b"x")
    assert str(err) == err.reason
