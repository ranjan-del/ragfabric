"""The Google Drive connector (Phase 10, Task 6).

Everything here runs against the real ``google-api-python-client`` with fake
HTTP (``HttpMockSequence``) and the static discovery document that ships inside
the library (design D24). No Google account, key or network is used.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from urllib.parse import parse_qs, urlparse

import pytest

pytest.importorskip("googleapiclient")

from googleapiclient.discovery import build  # noqa: E402
from googleapiclient.http import HttpMockSequence  # noqa: E402

from ragfabric_core.config_file import GoogleDriveConnectorConfig  # noqa: E402
from ragfabric_core.connectors import google_drive  # noqa: E402
from ragfabric_core.connectors.google_drive import (  # noqa: E402
    DOCX_MIME,
    FOLDER_MIME,
    LIST_FIELDS,
    PPTX_MIME,
    DriveSetupError,
    GoogleDriveConnector,
    build_drive_service,
)

ROOT = "root-folder"
OK = {"status": "200"}


def _page(files, next_token=None):
    body = {"files": files}
    if next_token:
        body["nextPageToken"] = next_token
    return (OK, json.dumps(body))


def _file(file_id, name, mime, **extra):
    return {"id": file_id, "name": name, "mimeType": mime, **extra}


def _pdf(file_id="f-pdf", name="leave.pdf"):
    return _file(
        file_id,
        name,
        "application/pdf",
        size="1234",
        md5Checksum="abc123",
        modifiedTime="2026-10-01T09:30:00.000Z",
        version="7",
    )


def _service(responses):
    http = HttpMockSequence(list(responses))
    service = build("drive", "v3", http=http, static_discovery=True, cache_discovery=False)
    return service, http


def _query(uri):
    return {k: v[0] for k, v in parse_qs(urlparse(uri).query).items()}


ROOT_FOLDER = (OK, json.dumps({"id": ROOT, "name": "Policies", "mimeType": FOLDER_MIME}))


def _connector(responses, **kwargs):
    """A connector whose first request (the root folder check) succeeds."""
    service, http = _service([ROOT_FOLDER, *responses])
    return GoogleDriveConnector(service, ROOT, **kwargs), http


def _listings(http):
    """The files.list requests, without the root folder check."""
    return [r for r in http.request_sequence if urlparse(r[0]).path.endswith("/drive/v3/files")]


# --- listing --------------------------------------------------------------------------


def test_list_sends_the_folder_query_fields_and_shared_drive_flags():
    connector, http = _connector([_page([_pdf()])])
    [source] = connector.list_documents()
    assert source.source_id == "f-pdf"
    [(uri, method, _body, _headers)] = _listings(http)
    assert method == "GET"
    assert urlparse(uri).path.endswith("/drive/v3/files")
    query = _query(uri)
    assert query["q"] == f"'{ROOT}' in parents and trashed = false"
    assert query["fields"] == LIST_FIELDS
    assert query["supportsAllDrives"] == "true"
    assert query["includeItemsFromAllDrives"] == "true"
    for field in ("id", "name", "mimeType", "size", "modifiedTime", "md5Checksum"):
        assert field in LIST_FIELDS
    assert "nextPageToken" in LIST_FIELDS


def test_list_follows_next_page_token():
    connector, http = _connector(
        [
            _page([_pdf("a", "a.pdf")], next_token="tok-2"),
            _page([_pdf("b", "b.pdf")]),
        ]
    )
    assert [s.source_id for s in connector.list_documents()] == ["a", "b"]
    first, second = (_query(r[0]) for r in _listings(http))
    assert "pageToken" not in first
    assert second["pageToken"] == "tok-2"
    assert second["q"] == first["q"]


def test_list_recurses_into_each_subfolder_once():
    sub = _file("sub", "Policies", FOLDER_MIME)
    connector, http = _connector(
        [
            # root holds the subfolder twice (a second parent), and a file
            _page([sub, _pdf("a", "a.pdf"), sub]),
            # the subfolder holds a file and points back at itself
            _page([_pdf("b", "b.pdf"), sub]),
        ]
    )
    assert sorted(s.source_id for s in connector.list_documents()) == ["a", "b"]
    queries = [_query(r[0])["q"] for r in _listings(http)]
    assert queries == [
        f"'{ROOT}' in parents and trashed = false",
        "'sub' in parents and trashed = false",
    ]


def test_non_recursive_does_not_descend():
    sub = _file("sub", "Policies", FOLDER_MIME)
    connector, http = _connector([_page([sub, _pdf()])], recursive=False)
    assert [s.source_id for s in connector.list_documents()] == ["f-pdf"]
    assert len(_listings(http)) == 1


def test_a_folder_id_with_a_quote_is_escaped_in_the_query():
    folder = (OK, json.dumps({"id": "it's", "name": "x", "mimeType": FOLDER_MIME}))
    service, http = _service([folder, _page([])])
    list(GoogleDriveConnector(service, "it's").list_documents())
    assert _query(_listings(http)[0][0])["q"] == "'it\\'s' in parents and trashed = false"


# --- mapping --------------------------------------------------------------------------


def test_a_pdf_keeps_its_name_and_its_md5_is_the_version():
    connector, _ = _connector([_page([_pdf()])])
    [source] = connector.list_documents()
    assert source.name == "leave.pdf"
    assert source.content_type == "application/pdf"
    assert source.version == "abc123"
    assert source.size_bytes == 1234
    assert source.modified_at == datetime(2026, 10, 1, 9, 30, tzinfo=UTC)
    assert source.ready is True


@pytest.mark.parametrize(
    ("mime", "name", "content_type"),
    [
        ("application/vnd.google-apps.document", "Handbook.docx", DOCX_MIME),
        ("application/vnd.google-apps.presentation", "Handbook.pptx", PPTX_MIME),
        ("application/vnd.google-apps.spreadsheet", "Handbook.csv", "text/csv"),
    ],
)
def test_google_native_files_become_exported_formats(mime, name, content_type):
    native = _file("g1", "Handbook", mime, modifiedTime="2026-10-02T00:00:00Z", version="42")
    connector, _ = _connector([_page([native])])
    [source] = connector.list_documents()
    assert source.name == name
    assert source.content_type == content_type
    # No md5 or size for a native file: the Drive version number stands in.
    assert source.size_bytes is None
    assert source.version == "v42"


@pytest.mark.parametrize(
    "mime",
    [
        "application/vnd.google-apps.shortcut",
        "application/vnd.google-apps.form",
        "application/vnd.google-apps.drawing",
        "image/png",
        "application/zip",
    ],
)
def test_shortcuts_forms_images_and_other_files_are_skipped_and_counted(mime):
    connector, _ = _connector([_page([_file("x", "thing.png", mime), _pdf()])])
    assert [s.source_id for s in connector.list_documents()] == ["f-pdf"]
    assert connector.ignored == [f"x (thing.png): {mime}"]


def test_a_supported_type_without_its_extension_gets_one():
    connector, _ = _connector([_page([_file("t", "notes", "text/plain", size="3")])])
    [source] = connector.list_documents()
    assert source.name == "notes.txt"


def test_a_file_named_with_a_supported_extension_is_kept_whatever_its_mime_type():
    connector, _ = _connector(
        [_page([_file("m", "readme.md", "application/octet-stream", size="3")])]
    )
    assert [s.name for s in connector.list_documents()] == ["readme.md"]


# --- fetching -------------------------------------------------------------------------


def test_fetch_of_a_binary_file_downloads_it_with_get_media():
    connector, http = _connector([_page([_pdf()]), (OK, b"%PDF-1.7 bytes")])
    list(connector.list_documents())
    assert connector.fetch("f-pdf") == b"%PDF-1.7 bytes"
    uri = http.request_sequence[-1][0]
    assert urlparse(uri).path.endswith("/drive/v3/files/f-pdf")
    query = _query(uri)
    assert query["alt"] == "media"
    assert query["supportsAllDrives"] == "true"


def test_fetch_of_a_google_doc_exports_it_as_docx():
    doc = _file("g1", "Handbook", "application/vnd.google-apps.document", version="1")
    connector, http = _connector([_page([doc]), (OK, b"PK docx bytes")])
    list(connector.list_documents())
    assert connector.fetch("g1") == b"PK docx bytes"
    uri = http.request_sequence[-1][0]
    assert urlparse(uri).path.endswith("/drive/v3/files/g1/export")
    assert _query(uri)["mimeType"] == DOCX_MIME


def test_fetch_of_an_unlisted_id_looks_up_its_type_first():
    sheet = _file("s1", "Budget", "application/vnd.google-apps.spreadsheet")
    service, http = _service([(OK, json.dumps(sheet)), (OK, b"a,b\n1,2\n")])
    connector = GoogleDriveConnector(service, ROOT)
    assert connector.fetch("s1") == b"a,b\n1,2\n"
    lookup, export = (r[0] for r in http.request_sequence)
    assert urlparse(lookup).path.endswith("/drive/v3/files/s1")
    assert _query(lookup)["fields"] == "id, name, mimeType"
    assert _query(export)["mimeType"] == "text/csv"


# --- setup errors ---------------------------------------------------------------------


def test_missing_extra_names_the_install_command(monkeypatch, tmp_path):
    key = tmp_path / "key.json"
    key.write_text("{}")
    for module in ("googleapiclient", "googleapiclient.discovery", "google.oauth2"):
        monkeypatch.setitem(sys.modules, module, None)
    with pytest.raises(DriveSetupError, match=r"pip install 'ragfabric\[drive\]'"):
        build_drive_service(str(key))


def test_a_missing_credentials_file_is_a_clear_error(tmp_path):
    missing = tmp_path / "nope.json"
    with pytest.raises(DriveSetupError, match="credentials file .*nope.json.* does not exist"):
        build_drive_service(str(missing))


@pytest.mark.parametrize(
    "content", ["not json", '{"type": "service_account"}', '{"type": "authorized_user"}']
)
def test_an_unreadable_credentials_file_is_a_clear_error_without_its_content(tmp_path, content):
    key = tmp_path / "key.json"
    key.write_text(content)
    with pytest.raises(DriveSetupError, match="not a readable service account key") as info:
        build_drive_service(str(key))
    assert content not in str(info.value)


def test_from_config_uses_the_key_and_the_read_only_scope(monkeypatch, tmp_path):
    seen = {}

    def fake_build(credentials_file):
        seen["file"] = credentials_file
        return "service"

    monkeypatch.setattr(google_drive, "build_drive_service", fake_build)
    cfg = GoogleDriveConnectorConfig(
        name="policies",
        kind="google_drive",
        folder_id="abc",
        credentials_file=str(tmp_path / "key.json"),
        recursive=False,
    )
    connector = GoogleDriveConnector.from_config(cfg)
    assert seen["file"] == str(tmp_path / "key.json")
    assert connector.folder_id == "abc"
    assert connector.recursive is False
    assert google_drive.SCOPES == ["https://www.googleapis.com/auth/drive.readonly"]


# --- the root folder ------------------------------------------------------------------


@pytest.mark.parametrize("status", ["404", "403"])
def test_a_wrong_or_unshared_folder_id_is_a_clear_error(status):
    """files.list on an unknown parent returns 200 with no files, which would look
    like an empty folder. The folder is looked up first so the operator is told."""
    error = json.dumps({"error": {"code": int(status), "message": "File not found: nope."}})
    service, _ = _service([({"status": status}, error)])
    with pytest.raises(DriveSetupError, match="not found or not shared with the service account"):
        list(GoogleDriveConnector(service, "nope").list_documents())


def test_a_folder_id_that_is_a_file_is_a_clear_error():
    service, _ = _service(
        [(OK, json.dumps({"id": "f", "name": "a.pdf", "mimeType": "application/pdf"}))]
    )
    with pytest.raises(DriveSetupError, match="is not a folder"):
        list(GoogleDriveConnector(service, "f").list_documents())
