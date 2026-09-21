"""Provider configuration read and write (Task 11).

This is the sensitive endpoint: it sits next to the OpenAI and Anthropic API
keys. The rules it has to hold are narrow and worth stating.

- The read reports which provider and model are selected and whether the
  secret that provider needs is present in the environment. It never returns
  the secret, and it never returns a masked form of it either: a mask that
  preserves length leaks length.
- Secrets stay in the environment, per ``ragfabric_core.config``. This
  endpoint selects and reports. It is not a secret store.
- Writing requires admin and is written to the audit log.
"""

from __future__ import annotations

import json

import pytest


@pytest.fixture()
def config_guard():
    """Restore ragfabric.yaml after a test that writes provider config.

    The configuration is a process wide singleton, so a test that switches
    the embedding provider would otherwise leak that choice into every test
    that runs after it in the same process.
    """
    from ragfabric_core import runtime
    from ragfabric_core.config_file import resolve_config_path

    path = resolve_config_path()
    assert path is not None, "the test session must point RAGFABRIC_CONFIG at a file"
    original = path.read_text(encoding="utf-8")
    yield path
    path.write_text(original, encoding="utf-8")
    runtime.reset_config()


def test_provider_config_never_returns_the_secret(client, admin_headers, config_guard, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-live-must-never-be-echoed-back")
    written = client.put(
        "/api/admin/providers",
        json={"llm": {"provider": "openai", "model": "gpt-5.4-mini", "base_url": None}},
        headers=admin_headers,
    )
    assert written.status_code == 200, written.text

    body = client.get("/api/admin/providers", headers=admin_headers).json()
    assert "sk-" not in json.dumps(body)
    assert body["llm"]["has_key"] is True
    assert body["llm"]["key_env_var"] == "OPENAI_API_KEY"
    assert body["llm"]["provider"] == "openai"


def test_a_provider_that_needs_no_key_reports_has_key_false(client, admin_headers, config_guard):
    body = client.get("/api/admin/providers", headers=admin_headers).json()
    assert body["embeddings"]["provider"] == "offline"
    assert body["embeddings"]["requires_key"] is False
    assert body["embeddings"]["has_key"] is False
    assert body["embeddings"]["key_env_var"] is None


def test_a_non_admin_cannot_read_or_write_provider_config(client, auth_headers):
    assert client.get("/api/admin/providers", headers=auth_headers).status_code == 403
    assert (
        client.put(
            "/api/admin/providers",
            json={"llm": {"provider": "offline"}},
            headers=auth_headers,
        ).status_code
        == 403
    )


def test_an_unknown_provider_is_a_422(client, admin_headers, config_guard):
    resp = client.put(
        "/api/admin/providers", json={"llm": {"provider": "nope"}}, headers=admin_headers
    )
    assert resp.status_code == 422, resp.text


def test_changing_the_embedding_model_requires_a_reindex(client, admin_headers, config_guard):
    resp = client.put(
        "/api/admin/providers",
        json={"embeddings": {"provider": "offline", "model": "something-else", "dim": 128}},
        headers=admin_headers,
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["requires_reindex"] is True


def test_changing_only_the_llm_does_not_require_a_reindex(client, admin_headers, config_guard):
    current = client.get("/api/admin/providers", headers=admin_headers).json()
    resp = client.put(
        "/api/admin/providers",
        json={
            "llm": {"provider": "offline", "model": "scripted", "base_url": None},
            "embeddings": {
                "provider": current["embeddings"]["provider"],
                "model": current["embeddings"]["model"],
                "dim": current["embeddings"]["dim"],
                "base_url": current["embeddings"]["base_url"],
            },
        },
        headers=admin_headers,
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["requires_reindex"] is False


def test_a_provider_write_is_audited(client, admin_headers, config_guard, db_session):
    from ragfabric_core.models.access import AuditLog

    client.put(
        "/api/admin/providers",
        json={"llm": {"provider": "offline", "model": "scripted"}},
        headers=admin_headers,
    )
    db_session.expire_all()
    rows = db_session.query(AuditLog).filter(AuditLog.action == "provider_config_update").all()
    assert len(rows) == 1
    assert rows[0].details["changed"]
    assert "OPENAI_API_KEY" not in json.dumps(rows[0].details)


def test_a_provider_write_persists_across_a_reload(client, admin_headers, config_guard):
    client.put(
        "/api/admin/providers",
        json={"llm": {"provider": "offline", "model": "scripted", "base_url": None}},
        headers=admin_headers,
    )
    from ragfabric_core.config_file import load_config

    reloaded = load_config(config_guard)
    assert reloaded.llm.provider == "offline"
    assert reloaded.llm.model == "scripted"
    # Writing the llm section must not disturb anything else in the file.
    assert reloaded.cache.kind == "memory"


def test_the_connection_test_reports_a_real_failure(
    client, admin_headers, config_guard, monkeypatch
):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client.put(
        "/api/admin/providers",
        json={"llm": {"provider": "openai", "model": "gpt-5.4-mini"}},
        headers=admin_headers,
    )
    resp = client.post("/api/admin/providers/test", json={"target": "llm"}, headers=admin_headers)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["ok"] is False
    assert body["detail"]
    assert "sk-" not in json.dumps(body)


def test_the_connection_test_reports_a_real_success(client, admin_headers, config_guard):
    resp = client.post(
        "/api/admin/providers/test", json={"target": "embeddings"}, headers=admin_headers
    )
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"target": "embeddings", "ok": True, "detail": "", "model": "hashing-128"}


def test_an_unknown_test_target_is_a_422(client, admin_headers):
    assert (
        client.post(
            "/api/admin/providers/test", json={"target": "database"}, headers=admin_headers
        ).status_code
        == 422
    )
