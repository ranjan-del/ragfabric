import socket
import sys

from ragfabric_core import diagnostics
from ragfabric_core.config_file import RagFabricConfig
from ragfabric_core.db import migrate
from ragfabric_core.providers.base import ProviderError


def _offline_cfg():
    return RagFabricConfig.model_validate(
        {"llm": {"provider": "offline"}, "embeddings": {"provider": "offline", "dim": 16}}
    )


def _ollama_cfg(dim=16):
    return RagFabricConfig.model_validate(
        {"llm": {"provider": "ollama"}, "embeddings": {"provider": "ollama", "dim": dim}}
    )


class _BrokenLLM:
    name = "broken"
    default_model = "x"

    def complete(self, *args, **kwargs):
        raise ProviderError("broken", "boom")


class _BrokenEmbedder:
    name = "broken"
    model = "x"
    dim = 16

    def embed(self, texts):
        raise ProviderError("broken", "boom")


class _ShortEmbedder:
    name = "short"
    model = "x"
    dim = 8

    def embed(self, texts):
        from ragfabric_core.providers.base import EmbeddingResult

        return EmbeddingResult(
            vectors=[[0.0] * 8], model="x", provider="short", input_tokens=1, latency_ms=0
        )


def test_python_check_passes_on_this_interpreter():
    result = diagnostics.check_python()
    assert result.status == "pass"
    assert f"{sys.version_info.major}.{sys.version_info.minor}" in result.detail


def test_config_is_warn_without_a_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RAGFABRIC_CONFIG", raising=False)
    assert diagnostics.check_config(None).status == "warn"


def test_config_fails_on_an_invalid_file_naming_the_key(tmp_path):
    p = tmp_path / "ragfabric.yaml"
    p.write_text("llm:\n  provider: nonsense\n")
    result = diagnostics.check_config(p)
    assert result.status == "fail"
    assert "llm.provider" in result.detail


def test_config_passes_on_a_valid_file(tmp_path):
    p = tmp_path / "ragfabric.yaml"
    p.write_text("llm:\n  provider: offline\n")
    assert diagnostics.check_config(p).status == "pass"


def test_database_passes_on_fresh_sqlite(tmp_path):
    assert diagnostics.check_database(f"sqlite:///{tmp_path / 'a.db'}").status == "pass"


def test_database_failure_never_leaks_the_password():
    result = diagnostics.check_database("postgresql+psycopg://rf:s3cret@127.0.0.1:1/rf")
    assert result.status == "fail"
    assert "s3cret" not in result.detail
    assert "s3cret" not in (result.fix or "")


def test_migrations_fail_before_upgrade_and_pass_after(tmp_path):
    url = f"sqlite:///{tmp_path / 'm.db'}"
    assert migrate.current_revision(url) is None
    assert diagnostics.check_migrations(url).status == "fail"
    migrate.upgrade(url)
    assert migrate.current_revision(url) == migrate.head_revision()
    assert diagnostics.check_migrations(url).status == "pass"


def test_migrations_failure_on_unreachable_database_hides_password():
    result = diagnostics.check_migrations("postgresql+psycopg://rf:s3cret@127.0.0.1:1/rf")
    assert result.status == "fail"
    assert "s3cret" not in result.detail


def test_llm_and_embeddings_skip_without_network():
    cfg = _ollama_cfg()
    for check in (diagnostics.check_llm, diagnostics.check_embeddings):
        result = check(cfg, network=False)
        assert result.status == "skip"
        assert result.detail


def test_llm_offline_is_a_labelled_warn():
    result = diagnostics.check_llm(_offline_cfg(), network=True)
    assert result.status == "warn"
    assert result.detail == "offline mode: answers are extractive"
    assert result.fix == (
        "install Ollama, or set OPENAI_API_KEY or ANTHROPIC_API_KEY, then "
        "mv ragfabric.yaml ragfabric.yaml.bak && ragfabric quickstart && ragfabric reindex --yes"
    )
    assert "--force" not in result.fix


def test_embeddings_offline_upgrade_never_suggests_force():
    result = diagnostics.check_embeddings(_offline_cfg(), network=True)
    assert result.status == "warn"
    assert "mv ragfabric.yaml ragfabric.yaml.bak && ragfabric quickstart" in result.fix
    assert "ragfabric reindex --yes" in result.fix
    assert "--force" not in result.fix


def test_embeddings_offline_is_a_labelled_warn():
    result = diagnostics.check_embeddings(_offline_cfg(), network=True)
    assert result.status == "warn"
    assert "offline" in result.detail


def test_llm_fails_when_the_provider_raises(monkeypatch):
    monkeypatch.setattr(diagnostics, "build_llm_provider", lambda cfg: _BrokenLLM())
    result = diagnostics.check_llm(_ollama_cfg(), network=True)
    assert result.status == "fail"
    assert "boom" in result.detail


def test_llm_makes_one_completion_of_at_most_one_token(monkeypatch):
    from ragfabric_core.providers.offline import ScriptedLLMProvider

    scripted = ScriptedLLMProvider(responses=["ok"])
    seen = {}
    original = scripted.complete

    def spy(messages, **kwargs):
        seen.update(kwargs)
        return original(messages, **kwargs)

    scripted.complete = spy
    monkeypatch.setattr(diagnostics, "build_llm_provider", lambda cfg: scripted)
    assert diagnostics.check_llm(_ollama_cfg(), network=True).status == "pass"
    assert scripted.calls == 1
    assert seen["max_tokens"] <= 1


def test_embeddings_fail_when_the_provider_raises(monkeypatch):
    monkeypatch.setattr(diagnostics, "build_embedding_provider", lambda cfg: _BrokenEmbedder())
    assert diagnostics.check_embeddings(_ollama_cfg(), network=True).status == "fail"


def test_embeddings_fail_on_a_dimension_mismatch(monkeypatch):
    monkeypatch.setattr(diagnostics, "build_embedding_provider", lambda cfg: _ShortEmbedder())
    result = diagnostics.check_embeddings(_ollama_cfg(dim=16), network=True)
    assert result.status == "fail"
    assert "8" in result.detail and "16" in result.detail


def test_embeddings_pass_when_dimension_matches(monkeypatch):
    monkeypatch.setattr(diagnostics, "build_embedding_provider", lambda cfg: _ShortEmbedder())
    assert diagnostics.check_embeddings(_ollama_cfg(dim=8), network=True).status == "pass"


def test_graph_is_skip_when_disabled_and_not_pass():
    assert diagnostics.check_graph(_offline_cfg()).status == "skip"


def test_graph_enabled_with_offline_llm_and_no_model_is_warn():
    cfg = RagFabricConfig.model_validate(
        {"llm": {"provider": "offline"}, "graph_store": {"enabled": True}}
    )
    result = diagnostics.check_graph(cfg)
    assert result.status == "warn"
    assert "graph_store.extraction_model" in (result.fix or "")


def test_graph_enabled_with_a_model_says_what_was_examined():
    cfg = RagFabricConfig.model_validate(
        {
            "llm": {"provider": "offline"},
            "graph_store": {"enabled": True, "extraction_model": "m-1"},
        }
    )
    result = diagnostics.check_graph(cfg)
    assert result.status == "pass"
    assert result.detail == "graph extraction configured with m-1; extraction not exercised"


def test_graph_enabled_with_a_real_llm_uses_the_provider_default():
    cfg = RagFabricConfig.model_validate({"graph_store": {"enabled": True}})
    assert diagnostics.check_graph(cfg).status == "pass"


def test_migrations_unknown_revision_says_upgrade_ragfabric(tmp_path):
    from sqlalchemy import create_engine, text

    url = f"sqlite:///{tmp_path / 'u.db'}"
    engine = create_engine(url)
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL)"))
        conn.execute(text("INSERT INTO alembic_version VALUES ('zzzz_from_the_future')"))
    engine.dispose()
    result = diagnostics.check_migrations(url)
    assert result.status == "fail"
    assert result.detail == (
        "database is at zzzz_from_the_future, which this version of ragfabric does not know"
    )
    assert result.fix == "upgrade ragfabric (pip install -U ragfabric)"


def test_migrations_behind_says_db_upgrade(tmp_path):
    result = diagnostics.check_migrations(f"sqlite:///{tmp_path / 'b.db'}")
    assert result.fix == "ragfabric db upgrade"


def test_llm_and_embeddings_catch_any_exception(monkeypatch):
    def boom(cfg):
        raise ValueError("bad thing at postgresql://u:pw@h/x")

    monkeypatch.setattr(diagnostics, "build_llm_provider", boom)
    monkeypatch.setattr(diagnostics, "build_embedding_provider", boom)
    for check in (diagnostics.check_llm, diagnostics.check_embeddings):
        result = check(_ollama_cfg(), network=True)
        assert result.status == "fail"
        assert result.detail.startswith("ValueError: bad thing")
        assert "pw" not in result.detail


def test_offline_warns_even_without_network():
    assert diagnostics.check_llm(_offline_cfg(), network=False).status == "warn"
    assert diagnostics.check_embeddings(_offline_cfg(), network=False).status == "warn"


def test_ollama_fixes_are_specific(monkeypatch):
    class Down:
        def complete(self, *a, **k):
            raise ProviderError("ollama", "Connection error.")

    monkeypatch.setattr(diagnostics, "build_llm_provider", lambda cfg: Down())
    assert diagnostics.check_llm(_ollama_cfg(), network=True).fix == "ollama serve"

    class Missing:
        def complete(self, *a, **k):
            raise ProviderError("ollama", "model 'llama3.2:3b' not found")

    monkeypatch.setattr(diagnostics, "build_llm_provider", lambda cfg: Missing())
    assert diagnostics.check_llm(_ollama_cfg(), network=True).fix == "ollama pull llama3.2:3b"


def test_server_warns_when_nothing_listens():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    result = diagnostics.check_server(f"http://127.0.0.1:{port}")
    assert result.status == "warn"
    assert "ragfabric serve" in (result.fix or "")


def test_mask_url_hides_the_password():
    assert diagnostics.mask_url("postgresql://u:pw@h/db") == "postgresql://u:***@h/db"
    assert diagnostics.mask_url("sqlite:///x.db") == "sqlite:///x.db"


def test_migrations_behind_carries_both_revisions():
    exc = diagnostics.MigrationsBehind(None, "abc")
    assert exc.current is None and exc.head == "abc"


def test_run_all_returns_every_check_and_skips_network_ones(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RAGFABRIC_CONFIG", raising=False)
    results = diagnostics.run_all(
        config_path=None,
        db_url=f"sqlite:///{tmp_path / 'r.db'}",
        server_url="http://127.0.0.1:1",
        network=False,
    )
    names = [r.name for r in results]
    assert len(names) == len(set(names)) >= 8
    by = {r.name: r for r in results}
    assert by["llm"].status != "pass" and by["embeddings"].status != "pass"
    assert by["server"].detail == "server probe skipped: network checks are off"
    assert by["migrations"].status == "fail"


def test_mask_url_masks_a_password_containing_an_at_sign():
    masked = diagnostics.mask_url("postgresql://u:p@ss@db:5432/x")
    assert masked == "postgresql://u:***@db:5432/x"
    assert "ss" not in masked.split("***")[0]


def test_mask_urls_in_masks_every_url_in_a_string():
    text = "tried postgresql://a:one@h1/x then postgresql://b:two@h2/y"
    masked = diagnostics.mask_urls_in(text)
    assert "one" not in masked and "two" not in masked
    assert masked == "tried postgresql://a:***@h1/x then postgresql://b:***@h2/y"


# final fix wave: C1 and I7, a client package that is not installed ------------


def _hide(monkeypatch, *packages):
    import importlib.util

    real = importlib.util.find_spec

    def fake(name, *args, **kwargs):
        return None if name.split(".")[0] in packages else real(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake)


def test_offline_fixes_start_with_the_pip_install_while_openai_is_missing(monkeypatch):
    _hide(monkeypatch, "openai")
    for check in (diagnostics.check_llm, diagnostics.check_embeddings):
        assert check(_offline_cfg(), network=True).fix.startswith(
            "pip install 'ragfabric[openai]', then "
        )


def test_a_provider_that_is_not_installed_gets_the_pip_fix(monkeypatch):
    class NotInstalled:
        def complete(self, *a, **k):
            raise ProviderError(
                "ollama",
                "openai is not installed. Install it with: pip install 'ragfabric[openai]'",
            )

    monkeypatch.setattr(diagnostics, "build_llm_provider", lambda cfg: NotInstalled())
    assert diagnostics.check_llm(_ollama_cfg(), network=True).fix == (
        "pip install 'ragfabric[openai]'"
    )


def test_the_provider_install_messages_say_pip_install(monkeypatch):
    from ragfabric_core.providers import anthropic_provider, openai_compat

    def missing():
        raise ImportError("no")

    monkeypatch.setattr(openai_compat, "_import_openai", missing)
    monkeypatch.setattr(anthropic_provider, "_import_anthropic", missing)
    messages = []
    for build in (
        lambda: openai_compat._build_client("ollama", "k", None),
        lambda: anthropic_provider.AnthropicProvider(api_key="k"),
    ):
        try:
            build()
        except ProviderError as exc:
            messages.append(str(exc))
    assert len(messages) == 2
    assert all("pip install 'ragfabric[" in m and "uv pip" not in m for m in messages)


# final fix wave: I1, mask_url with '/', '@' and ':' in the password -----------


def test_mask_url_masks_a_password_containing_a_slash():
    masked = diagnostics.mask_url("postgresql+psycopg://rf:ab/cd@db.example.com:5432/rag")
    assert masked == "postgresql+psycopg://rf:***@db.example.com:5432/rag"


def test_mask_url_masks_a_password_with_slash_at_and_colon():
    masked = diagnostics.mask_url("postgresql://rf:a:b@c/d?e#f@db:5432/rag?sslmode=require")
    assert masked == "postgresql://rf:***@db:5432/rag?sslmode=require"


def test_mask_url_leaves_a_url_without_a_password_alone():
    for url in ("postgresql://rf@db:5432/rag", "http://127.0.0.1:8000/health", "sqlite:///./x.db"):
        assert diagnostics.mask_url(url) == url


def test_mask_urls_in_masks_two_urls_with_slashes_in_their_passwords():
    text = "a postgresql://u:p/1@h1/x b redis://:s/2@h2:6379/0"
    assert diagnostics.mask_urls_in(text) == "a postgresql://u:***@h1/x b redis://:***@h2:6379/0"


# final fix wave: I2, a server that is not RagFabric ----------------------------


def _serve_once(status: int, body: bytes):
    """A tiny HTTP server in a thread answering every GET with ``status`` and ``body``."""
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 - the http.server API
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def _check_against(status: int, body: bytes):
    server = _serve_once(status, body)
    try:
        return diagnostics.check_server(f"http://127.0.0.1:{server.server_address[1]}")
    finally:
        server.shutdown()
        server.server_close()


def test_server_passes_only_when_health_identifies_ragfabric():
    result = _check_against(200, b'{"status": "ok", "service": "ragfabric"}')
    assert result.status == "pass"


def test_a_2xx_health_from_another_program_is_not_ragfabric():
    result = _check_against(200, b'{"status": "ok"}')
    assert result.status == "warn"
    assert result.detail.endswith("is not a RagFabric server (HTTP 200 on /health)")
    assert "ragfabric serve --port " in result.fix and "in .env" in result.fix


def test_a_404_health_is_not_ragfabric():
    result = _check_against(404, b'{"detail": "Not Found"}')
    assert result.status == "warn"
    assert result.detail.endswith("is not a RagFabric server (HTTP 404 on /health)")
