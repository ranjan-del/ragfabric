import pytest
from pydantic import ValidationError

from ragfabric_core.config_file import RagFabricConfig, load_config


def test_defaults_when_no_file(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RAGFABRIC_CONFIG", raising=False)
    cfg = load_config()
    assert isinstance(cfg, RagFabricConfig)
    assert cfg.llm.provider == "openai" and cfg.vector_store.kind == "pgvector"
    assert cfg.graph_store.enabled is False and cfg.ingestion.chunk_size == 600
    assert cfg.strategies.traditional["top_k"] == 8 and cfg.router.mode == "auto"


def test_explicit_path_and_partial_override(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("llm:\n  provider: ollama\n  model: qwen3:8b\nvector_store:\n  kind: chroma\n")
    cfg = load_config(p)
    assert cfg.llm.provider == "ollama" and cfg.llm.model == "qwen3:8b"
    assert cfg.vector_store.kind == "chroma" and cfg.embeddings.provider == "openai"


def test_env_var_locates_the_file(tmp_path, monkeypatch):
    p = tmp_path / "x.yaml"
    p.write_text("router:\n  mode: manual\n")
    monkeypatch.setenv("RAGFABRIC_CONFIG", str(p))
    assert load_config().router.mode == "manual"


def test_cwd_ragfabric_yaml_is_found(tmp_path, monkeypatch):
    (tmp_path / "ragfabric.yaml").write_text("cache:\n  kind: memory\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RAGFABRIC_CONFIG", raising=False)
    assert load_config().cache.kind == "memory"


def test_unknown_keys_and_bad_values_fail_loudly(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("llm:\n  provider: openai\n  modle: typo\n")
    with pytest.raises(ValidationError):
        load_config(p)
    p.write_text("vector_store:\n  kind: pinecone\n")
    with pytest.raises(ValidationError):
        load_config(p)


def test_example_file_in_repo_root_is_valid():
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    cfg = load_config(root / "ragfabric.example.yaml")
    assert cfg.llm.provider in {"openai", "anthropic", "ollama", "offline"}
