"""FastAPI application entrypoint for RagFabric.

Wires the API routers together, configures CORS for the Angular frontend, and on
startup: creates the database tables, seeds a bootstrap admin (if configured), and
rebuilds the in-memory vector index from the persisted chunk embeddings so search
works immediately after a restart.

The default configuration is fully offline: SQLite database, deterministic
hashing embedder, in-memory vector store, and an extractive answer generator. No
API key or external service is required to run or test the app.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ragfabric_core.config import get_settings
from ragfabric_core.db.session import SessionLocal, init_db
from ragfabric_core.models.user import Role, User
from ragfabric_core.security import hash_password
from ragfabric_core.store.vector_store import get_store
from ragfabric_server.api.routes import (
    access,
    admin,
    analytics,
    auth,
    collections,
    documents,
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
    """Startup/shutdown: prepare the database and warm the vector index.

    The rebuild is what makes a restart safe. Vectors live in process memory, so
    without reloading them from ``chunks.embedding`` here, every document
    uploaded before the restart would still be listed in the UI but would be
    invisible to search.
    """
    init_db()
    _seed_admin()
    with SessionLocal() as db:
        restored = get_store().rebuild_from_db(db)
    logger.info("vector index warm: %d chunk vectors restored from the database", restored)
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
app.include_router(runs.router, prefix="/api/runs", tags=["runs"])


@app.get("/health", tags=["system"])
def health() -> dict:
    """Liveness/readiness probe used by Docker and the hosting platform.

    Reports the live vector-index size too, which is the cheapest way to confirm
    from outside the process that the startup rebuild actually ran.
    """
    return {
        "status": "ok",
        "service": settings.app_name,
        "version": settings.version,
        "index": get_store().stats(),
    }
