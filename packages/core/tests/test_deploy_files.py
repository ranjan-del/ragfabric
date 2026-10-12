"""The deployment files say what the guides say they do (Phase 10, Task 8).

Parsed, not run: ``docker compose config`` and a real start are the manual
checks in docs/learning/hardening-first-run.md. What is asserted here is what a
careless edit would break silently: only Caddy publishes ports, production
safety rails and JSON logs are on, the configs validate, and the Render
blueprint points at a Dockerfile that exists.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from ragfabric_core.config import Settings
from ragfabric_core.config_file import RagFabricConfig

ROOT = Path(__file__).resolve().parents[3]
COMPOSE = ROOT / "deploy" / "compose"


@pytest.fixture(scope="module")
def prod():
    return yaml.safe_load((COMPOSE / "docker-compose.prod.yml").read_text())


def _env(service: dict) -> dict:
    env = service.get("environment", {})
    if isinstance(env, list):
        env = dict(item.split("=", 1) for item in env)
    return env


def test_only_caddy_publishes_ports(prod):
    published = {name for name, svc in prod["services"].items() if svc.get("ports")}
    assert published == {"caddy"}
    for name, svc in prod["services"].items():
        assert "network_mode" not in svc, name


def test_the_expected_services_and_profiles_exist(prod):
    services = prod["services"]
    assert {"caddy", "api", "ui", "postgres", "redis"} <= set(services)
    for always_on in ("caddy", "api", "ui", "postgres", "redis"):
        assert "profiles" not in services[always_on], always_on
    assert services["connectors"]["profiles"] == ["connectors"]
    assert services["ollama"]["profiles"] == ["ollama"]
    assert services["worker"]["profiles"] == ["workers"]


@pytest.mark.parametrize("service", ["api", "worker", "connectors"])
def test_app_containers_run_in_production_with_json_logs(prod, service):
    env = _env(prod["services"][service])
    assert env["ENVIRONMENT"] == "production"
    assert env["RAGFABRIC_LOG_FORMAT"] == "json"
    assert env["RAGFABRIC_CONFIG"] == "/app/ragfabric.yaml"
    # Required, never defaulted: compose refuses to start without them.
    assert ":?" in env["JWT_SECRET"]
    assert ":?" in prod["services"]["postgres"]["environment"]["POSTGRES_PASSWORD"]


def test_the_api_trusts_forwarded_headers_only_because_it_is_unreachable(prod):
    api = prod["services"]["api"]
    assert _env(api)["FORWARDED_ALLOW_IPS"] == "*"
    assert not api.get("ports")


def test_the_connectors_service_watches_the_inbox_and_shares_uploads(prod):
    connectors = prod["services"]["connectors"]
    assert connectors["command"][:3] == ["ragfabric", "connectors", "sync"]
    assert "--watch" in connectors["command"]
    mounts = " ".join(str(v) for v in connectors["volumes"])
    assert "/data/inbox" in mounts and "uploads:/app/data/uploads" in mounts


def test_the_caddyfile_routes_the_api_and_the_ui():
    caddyfile = (COMPOSE / "Caddyfile").read_text()
    assert "reverse_proxy api:8000" in caddyfile
    assert "reverse_proxy ui:80" in caddyfile
    assert "/api/*" in caddyfile and "/health" in caddyfile


def test_the_prod_config_validates_and_selects_json_logs_and_redis():
    data = yaml.safe_load((COMPOSE / "ragfabric.prod.yaml").read_text())
    cfg = RagFabricConfig.model_validate(data)
    assert cfg.logging.format == "json"
    assert cfg.cache.kind == "redis"
    assert cfg.vector_store.kind == "pgvector"
    assert [c.name for c in cfg.connectors] == ["inbox"]
    assert cfg.connectors[0].path == "/data/inbox"


def test_the_env_example_holds_no_usable_secret():
    lines = (COMPOSE / ".env.prod.example").read_text().splitlines()
    values = dict(line.split("=", 1) for line in lines if "=" in line and not line.startswith("#"))
    for key in ("JWT_SECRET", "POSTGRES_PASSWORD", "FIRST_ADMIN_PASSWORD"):
        assert values[key] == "", key
    assert "DOMAIN" in values


def test_render_blueprint_parses_and_builds_the_api_dockerfile():
    blueprint = yaml.safe_load((ROOT / "render.yaml").read_text())
    [api] = [s for s in blueprint["services"] if s["name"] == "ragfabric-api"]
    assert api["runtime"] == "docker"
    assert (ROOT / api["dockerfilePath"]).is_file()
    assert api["healthCheckPath"] == "/health"
    env = {e["key"]: e for e in api["envVars"]}
    assert env["ENVIRONMENT"]["value"] == "production"
    assert env["JWT_SECRET"]["generateValue"] is True
    assert "fromDatabase" in env["DATABASE_URL"]
    assert blueprint["databases"][0]["name"]


@pytest.mark.parametrize(
    "url",
    [
        "postgres://u:p@db:5432/rag",
        "postgresql://u:p@db:5432/rag",
        "postgresql+psycopg://u:p@db:5432/rag",
    ],
)
def test_a_hosted_postgres_url_selects_the_installed_driver(monkeypatch, url):
    """Render and Railway hand out postgres:// or postgresql:// URLs, which
    SQLAlchemy maps to psycopg2; RagFabric ships psycopg 3."""
    monkeypatch.setenv("DATABASE_URL", url)
    assert Settings().database_url == "postgresql+psycopg://u:p@db:5432/rag"


def test_a_sqlite_url_is_left_alone(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "sqlite:///./x.db")
    assert Settings().database_url == "sqlite:///./x.db"


def test_the_render_config_validates_and_keeps_the_pinned_width():
    data = yaml.safe_load((ROOT / "deploy" / "render" / "ragfabric.yaml").read_text())
    cfg = RagFabricConfig.model_validate(data)
    assert cfg.embeddings.dim == 768
    assert cfg.logging.format == "json"
