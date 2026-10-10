"""The watched folder connector (Phase 10, Task 5)."""

from __future__ import annotations

import os
from datetime import UTC, datetime

import pytest

from ragfabric_core.connectors.folder import FolderConnector

NOW = 1_800_000_000.0


def _write(path, content=b"annual leave", age=60.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    os.utime(path, (NOW - age, NOW - age))
    return path


def _connector(root, **kwargs):
    return FolderConnector(root, clock=lambda: NOW, **kwargs)


def _ids(connector):
    return [s.source_id for s in connector.list_documents()]


def test_lists_supported_files_recursively_with_relative_ids(tmp_path):
    _write(tmp_path / "a.txt")
    _write(tmp_path / "policies" / "leave.pdf")
    _write(tmp_path / "policies" / "image.png")
    sources = {s.source_id: s for s in _connector(tmp_path).list_documents()}
    assert sorted(sources) == ["a.txt", "policies/leave.pdf"]
    leave = sources["policies/leave.pdf"]
    assert leave.name == "leave.pdf"
    assert leave.size_bytes == len(b"annual leave")
    assert leave.modified_at == datetime.fromtimestamp(NOW - 60.0, UTC)
    assert leave.ready is True


def test_non_recursive_lists_only_the_top_level(tmp_path):
    _write(tmp_path / "a.txt")
    _write(tmp_path / "sub" / "b.txt")
    assert _ids(_connector(tmp_path, recursive=False)) == ["a.txt"]


@pytest.mark.parametrize(
    "name",
    [".hidden.txt", "~$draft.docx", "upload.txt.part", "file.crdownload", "x.tmp", ".git/c.md"],
)
def test_hidden_and_temporary_files_are_ignored(tmp_path, name):
    _write(tmp_path / name)
    _write(tmp_path / "keep.txt")
    assert _ids(_connector(tmp_path)) == ["keep.txt"]


def test_a_symlink_that_leaves_the_folder_is_ignored(tmp_path):
    outside = tmp_path / "outside"
    inbox = tmp_path / "inbox"
    _write(outside / "secret.txt")
    _write(inbox / "keep.txt")
    (inbox / "link.txt").symlink_to(outside / "secret.txt")
    (inbox / "linkdir").symlink_to(outside, target_is_directory=True)
    assert _ids(_connector(inbox)) == ["keep.txt"]


def test_a_symlink_inside_the_folder_is_followed(tmp_path):
    _write(tmp_path / "real.txt")
    (tmp_path / "alias.txt").symlink_to(tmp_path / "real.txt")
    assert _ids(_connector(tmp_path)) == ["alias.txt", "real.txt"]


def test_a_file_still_being_written_is_listed_as_not_ready(tmp_path):
    _write(tmp_path / "fresh.txt", age=1.0)
    _write(tmp_path / "old.txt", age=60.0)
    ready = {s.source_id: s.ready for s in _connector(tmp_path, settle_seconds=5).list_documents()}
    assert ready == {"fresh.txt": False, "old.txt": True}


def test_fetch_reads_the_bytes_and_refuses_paths_outside_the_folder(tmp_path):
    _write(tmp_path / "inbox" / "a.txt", b"hello")
    _write(tmp_path / "secret.txt", b"no")
    connector = _connector(tmp_path / "inbox")
    assert connector.fetch("a.txt") == b"hello"
    with pytest.raises(PermissionError):
        connector.fetch("../secret.txt")


def test_a_missing_folder_raises_so_nothing_is_deleted(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        list(_connector(tmp_path / "nope").list_documents())
