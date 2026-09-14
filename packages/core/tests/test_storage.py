from pathlib import Path

import pytest

from ragfabric_core.ingest.storage import FileStorage


def test_save_writes_under_document_id_and_sanitises_the_name(tmp_path):
    fs = FileStorage(tmp_path)
    rel = fs.save(7, "../../etc/passwd weird name.PDF", b"%PDF-1.4")
    assert rel == "7/etc_passwd_weird_name.PDF"
    assert fs.path_for(rel).read_bytes() == b"%PDF-1.4"
    assert fs.path_for(rel).resolve().is_relative_to(tmp_path.resolve())


def test_long_names_are_capped_but_keep_the_extension(tmp_path):
    fs = FileStorage(tmp_path)
    rel = fs.save(9, "a" * 300 + ".pdf", b"x")
    name = rel.split("/", 1)[1]
    assert len(name) <= 120 and name.endswith(".pdf") and name.startswith("aaa")
    assert fs.path_for(rel).read_bytes() == b"x"
    rel2 = fs.save(9, "b" * 300, b"y")
    assert len(rel2.split("/", 1)[1]) <= 120


def test_delete_removes_the_document_directory(tmp_path):
    fs = FileStorage(tmp_path)
    fs.save(3, "a.txt", b"a")
    fs.delete(3)
    assert not (tmp_path / "3").exists()
    fs.delete(3)  # idempotent


def test_path_for_rejects_escapes(tmp_path):
    fs = FileStorage(tmp_path)
    with pytest.raises(ValueError):
        fs.path_for("../outside")


def test_get_storage_uses_config(tmp_path, monkeypatch):
    from ragfabric_core import runtime
    from ragfabric_core.ingest.storage import get_storage

    p = tmp_path / "ragfabric.yaml"
    p.write_text(f"ingestion:\n  uploads_dir: {tmp_path / 'blobs'}\n")
    monkeypatch.setenv("RAGFABRIC_CONFIG", str(p))
    runtime.reset_config()
    assert get_storage().root == Path(tmp_path / "blobs")
    runtime.reset_config()
