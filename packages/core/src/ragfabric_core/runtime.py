"""Process wide accessors for configuration and the database session factory.

`get_config()` loads ragfabric.yaml once (path from RAGFABRIC_CONFIG or the
working directory) and caches it; `reset_config()` exists for tests and for the
CLI after `ragfabric init` writes a new file. Keeping these here, rather than
importing module level singletons everywhere, lets tests swap configuration
without reloading modules.
"""

from __future__ import annotations

from ragfabric_core.config_file import RagFabricConfig, load_config

_config: RagFabricConfig | None = None


def get_config() -> RagFabricConfig:
    global _config
    if _config is None:
        _config = load_config()
    return _config


def reset_config() -> None:
    global _config
    _config = None


def get_session_factory():
    from ragfabric_core.db.session import SessionLocal

    return SessionLocal
