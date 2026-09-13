from ragfabric_core import runtime
from ragfabric_core.config_file import IngestionConfig


def test_ingestion_config_new_fields_have_defaults():
    cfg = IngestionConfig()
    assert cfg.uploads_dir == "data/uploads"
    assert cfg.indexing == "inline"


def test_get_config_is_cached_and_resettable(tmp_path, monkeypatch):
    p = tmp_path / "ragfabric.yaml"
    p.write_text("ingestion:\n  indexing: queue\n  chunk_size: 300\n")
    monkeypatch.setenv("RAGFABRIC_CONFIG", str(p))
    runtime.reset_config()
    first = runtime.get_config()
    assert first.ingestion.indexing == "queue" and first.ingestion.chunk_size == 300
    p.write_text("ingestion:\n  indexing: inline\n")
    assert runtime.get_config() is first
    runtime.reset_config()
    assert runtime.get_config().ingestion.indexing == "inline"


def test_session_factory_is_the_shared_one():
    from ragfabric_core.db.session import SessionLocal

    assert runtime.get_session_factory() is SessionLocal
