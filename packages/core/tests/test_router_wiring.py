"""The router's settings, and proof that every one of them is read by something."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ragfabric_core.config_file import RagFabricConfig, RouterConfig
from ragfabric_core.models import Base
from ragfabric_core.router.mode import resolve_requested
from ragfabric_core.strategies.base import StrategyName as S
from ragfabric_core.strategies.registry_defaults import default_registry


@pytest.fixture()
def session_factory(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'wiring.db'}")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)


def test_unset_resolves_from_the_router_mode():
    assert resolve_requested(None, RouterConfig(mode="auto")) is S.AUTO
    assert resolve_requested(None, RouterConfig(mode="manual")) is S.TRADITIONAL


@pytest.mark.parametrize("mode", ["auto", "manual"])
def test_a_named_strategy_always_bypasses_the_router(mode):
    assert resolve_requested("graph", RouterConfig(mode=mode)) is S.GRAPH


def test_every_router_setting_reaches_something(session_factory):
    cfg = RagFabricConfig.model_validate(
        {
            "llm": {"provider": "offline"},
            "embeddings": {"provider": "offline", "dim": 768},
            "router": {"mode": "manual", "min_confidence": 0.33, "classifier_model": "tiny"},
            "graph_store": {"enabled": True},
        }
    )
    auto = default_registry(cfg, session_factory).get(S.AUTO)
    checks = {
        "mode": lambda: resolve_requested(None, cfg.router) is S.TRADITIONAL,
        "min_confidence": lambda: auto.min_confidence == 0.33,
        "classifier_model": lambda: auto.classifier_model == "tiny",
    }
    assert set(checks) == set(RouterConfig.model_fields), "a RouterConfig field is not checked"
    assert all(check() for check in checks.values())


def test_classifier_model_null_uses_the_llm_model(session_factory):
    cfg = RagFabricConfig.model_validate(
        {
            "llm": {"provider": "offline", "model": "m1"},
            "embeddings": {"provider": "offline", "dim": 768},
        }
    )
    assert default_registry(cfg, session_factory).get(S.AUTO).classifier_model == "m1"
