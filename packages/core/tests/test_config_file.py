import pytest
from pydantic import ValidationError

from ragfabric_core.config_file import (
    LexicalStoreConfig,
    RagFabricConfig,
    VectorlessConfig,
    load_config,
)


def test_defaults_when_no_file(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RAGFABRIC_CONFIG", raising=False)
    cfg = load_config()
    assert isinstance(cfg, RagFabricConfig)
    assert cfg.llm.provider == "ollama" and cfg.vector_store.kind == "pgvector"
    assert cfg.graph_store.enabled is False and cfg.ingestion.chunk_size == 600
    assert cfg.strategies.traditional["top_k"] == 8 and cfg.router.mode == "auto"


def test_explicit_path_and_partial_override(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("llm:\n  provider: ollama\n  model: qwen3:8b\nvector_store:\n  kind: chroma\n")
    cfg = load_config(p)
    assert cfg.llm.provider == "ollama" and cfg.llm.model == "qwen3:8b"
    assert cfg.vector_store.kind == "chroma" and cfg.embeddings.provider == "ollama"


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


def test_code_defaults_with_no_config_file(monkeypatch, tmp_path):
    """Verify that code defaults match the no-key-first-run promise: Ollama."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RAGFABRIC_CONFIG", raising=False)
    cfg = load_config()
    assert cfg.llm.provider == "ollama"
    assert cfg.llm.model == "llama3.2:3b"
    assert cfg.llm.base_url == "http://localhost:11434/v1"
    assert cfg.embeddings.provider == "ollama"
    assert cfg.embeddings.model == "nomic-embed-text"
    assert cfg.embeddings.dim == 768
    assert cfg.embeddings.base_url == "http://localhost:11434/v1"


def test_default_example_config_is_the_no_key_ollama_path(tmp_path):
    from pathlib import Path

    from ragfabric_core.config_file import load_config

    example = Path(__file__).resolve().parents[3] / "ragfabric.example.yaml"
    cfg = load_config(example)
    assert cfg.embeddings.provider == "ollama"
    assert cfg.embeddings.model == "nomic-embed-text"
    assert cfg.embeddings.dim == 768
    assert cfg.llm.provider == "ollama"
    assert cfg.llm.model == "llama3.2:3b"
    assert cfg.llm.base_url == "http://localhost:11434/v1"


# --- Task 9: the vectorless strategy configuration -------------------------
#
# StrategiesConfig.vectorless used to be dict[str, float | int | str | bool].
# That dict accepted any key, so `k_1: 1.2` was stored and never read, which
# contradicts this file's own stated principle that a typo which silently
# falls back to a default is the worst kind of configuration bug.


def test_an_unknown_vectorless_key_is_rejected():
    with pytest.raises(ValidationError):
        VectorlessConfig(k_1=1.2)  # typo


def test_b_outside_zero_to_one_is_rejected():
    with pytest.raises(ValidationError):
        VectorlessConfig(b=1.5)


def test_b_below_zero_is_rejected():
    # Negative b rewards long chunks, which is not "less normalisation", it is
    # the opposite of normalisation.
    with pytest.raises(ValidationError):
        VectorlessConfig(b=-0.1)


def test_a_boost_below_one_is_rejected():
    with pytest.raises(ValidationError):
        VectorlessConfig(phrase_boost=0.5)


def test_an_identifier_boost_below_one_is_rejected():
    with pytest.raises(ValidationError):
        VectorlessConfig(identifier_boost=0.9)


def test_defaults_match_the_documented_bm25_defaults():
    c = VectorlessConfig()
    assert (c.k1, c.b, c.fusion_k) == (1.2, 0.75, 60)


def test_the_remaining_vectorless_defaults():
    c = VectorlessConfig()
    assert c.top_k == 8
    assert (c.phrase_boost, c.identifier_boost) == (2.0, 3.0)
    assert c.fusion_weights == (1.0, 1.0)
    assert c.max_context_tokens == 6000


def test_the_vectorless_block_loads_from_yaml(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("strategies:\n  vectorless:\n    k1: 1.5\n    b: 0.0\n    fusion_k: 10\n")
    cfg = load_config(p)
    assert (cfg.strategies.vectorless.k1, cfg.strategies.vectorless.b) == (1.5, 0.0)
    assert cfg.strategies.vectorless.fusion_k == 10


def test_a_typo_in_the_vectorless_block_fails_the_file_not_just_the_model(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("strategies:\n  vectorless:\n    k_1: 1.5\n")
    with pytest.raises(ValidationError):
        load_config(p)


def test_the_lexical_store_accepts_the_in_process_kind_and_a_cap():
    cfg = LexicalStoreConfig(kind="bm25_memory", max_chunks=1000)
    assert cfg.kind == "bm25_memory" and cfg.max_chunks == 1000


def test_an_unknown_lexical_store_kind_is_rejected():
    with pytest.raises(ValidationError):
        LexicalStoreConfig(kind="elasticsearch")
