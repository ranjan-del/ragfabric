"""FastAPI application entrypoint for RagFabric.

Wires the API routers together, configures CORS for the Angular frontend, and on
startup creates the database tables and seeds a bootstrap admin (if configured).

The default configuration is fully offline: SQLite database, deterministic
hashing embedder, and an extractive answer generator. No API key or external
service is required to run or test the app.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ragfabric_core.config import get_settings
from ragfabric_core.db.session import SessionLocal, init_db
from ragfabric_core.models.user import Role, User
from ragfabric_core.providers.registry import build_embedding_provider, build_llm_provider
from ragfabric_core.runtime import get_config, get_session_factory
from ragfabric_core.security import hash_password
from ragfabric_core.stores.registry import build_cache, build_lexical_store, build_vector_store
from ragfabric_core.strategies.registry_defaults import default_registry
from ragfabric_core.telemetry.tracing import configure_otel
from ragfabric_server.api.routes import (
    access,
    admin,
    analytics,
    ask,
    auth,
    collections,
    documents,
    providers,
    runs,
    search,
)

settings = get_settings()
logger = logging.getLogger(__name__)


def _seed_admin() -> None:
    """Create the bootstrap admin account once, if configured and absent."""
    if not (settings.first_admin_email and settings.first_admin_password):
        return
    with SessionLocal() as db:
        email = settings.first_admin_email.lower()
        if db.query(User).filter(User.email == email).first() is not None:
            return
        db.add(
            User(
                email=email,
                hashed_password=hash_password(settings.first_admin_password),
                role=Role.ADMIN.value,
            )
        )
        db.commit()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown: prepare the database and the process wide singletons.

    Task 16 moves ``deps.py``'s cache, LLM provider, strategy registry,
    vector store and lexical store off module-level globals built lazily on
    first use, and onto ``app.state``, built exactly once, right here, at
    application startup. ``deps.get_reranker`` is deliberately not among
    them; see its own docstring in ``deps.py`` for why moving it would
    ripple past this task.
    """
    init_db()
    export_on = configure_otel(get_config().telemetry.otlp_endpoint)
    logger.info("OTLP export %s", "enabled" if export_on else "disabled")
    _seed_admin()

    cfg = get_config()
    session_factory = get_session_factory()
    app.state.cache = build_cache(cfg.cache)
    app.state.llm_provider = build_llm_provider(cfg.llm)
    app.state.strategy_registry = default_registry(cfg, session_factory)
    embedder = build_embedding_provider(cfg.embeddings)
    app.state.vector_store = build_vector_store(
        cfg.vector_store, session_factory, embedding_model=embedder.model
    )
    app.state.lexical_store = build_lexical_store(cfg.lexical_store, session_factory)

    yield


app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description="RagFabric API: self hosted, measurement first retrieval augmented generation.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routers. See MEMORY.md "Dashboard" / "Admin" sections.
app.include_router(auth.router, prefix="/api/auth", tags=["auth"])
app.include_router(documents.router, prefix="/api/documents", tags=["documents"])
app.include_router(search.router, prefix="/api/search", tags=["search"])
app.include_router(collections.router, prefix="/api/collections", tags=["collections"])
app.include_router(analytics.router, prefix="/api/analytics", tags=["analytics"])
app.include_router(admin.router, prefix="/api/admin", tags=["admin"])
app.include_router(access.router, prefix="/api/admin", tags=["access"])
app.include_router(providers.router, prefix="/api/admin", tags=["providers"])
app.include_router(runs.router, prefix="/api/runs", tags=["runs"])
app.include_router(ask.router, prefix="/api", tags=["ask"])


@app.get("/health", tags=["system"])
def health() -> dict:
    """Liveness/readiness probe used by Docker and the hosting platform.

    Reports the live vector count from the configured store too, a cheap way
    to confirm from outside the process that the store is reachable.
    """
    cfg = get_config()
    store = build_vector_store(cfg.vector_store, get_session_factory())
    return {
        "status": "ok",
        "service": settings.app_name,
        "version": settings.version,
        "index": {"vectors": store.count()},
    }
