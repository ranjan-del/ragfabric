"""Shared FastAPI dependencies: current-user resolution and role guards.

- ``get_current_user`` decodes the bearer access token and loads the user.
- ``require_role("admin")`` builds a dependency that also checks the role.
- ``get_principal`` resolves the caller as a Principal: an API key (header or
  bearer) or a signed in user, so route handlers can depend on one thing
  regardless of how the caller authenticated.
- ``get_access_filter`` computes the AccessFilter for that principal.
- ``get_cache`` builds the process wide Cache once from configuration.

401 Unauthorized -> we don't know who you are (bad/missing/expired token).
403 Forbidden    -> we know who you are, but you're not allowed.
429 Too Many Requests -> an API key went over its rate limit.
"""

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from ragfabric_core.auth.api_keys import (
    KEY_PREFIX,
    principal_for_api_key,
    principal_for_user,
    verify_api_key,
)
from ragfabric_core.auth.policy import compute_access_filter
from ragfabric_core.auth.principal import AccessFilter, Principal
from ragfabric_core.auth.ratelimit import check_rate_limit
from ragfabric_core.db.session import get_db
from ragfabric_core.models.user import User
from ragfabric_core.providers.base import LLMProvider
from ragfabric_core.rerank.base import Reranker
from ragfabric_core.runtime import get_config, get_session_factory
from ragfabric_core.security import ACCESS, JWTError, decode_token
from ragfabric_core.stores.base import Cache, LexicalStore, VectorStore
from ragfabric_core.stores.registry import build_cache
from ragfabric_core.strategies.base import StrategyRegistry

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/auth/login", auto_error=False)

_credentials_error = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Could not validate credentials",
    headers={"WWW-Authenticate": "Bearer"},
)

_cache: Cache | None = None


def get_current_user(
    token: str | None = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:
    """Return the active user for the request's access token."""
    if not token:
        raise _credentials_error
    try:
        payload = decode_token(token)
    except JWTError as exc:
        raise _credentials_error from exc

    if payload.get("type") != ACCESS:
        raise _credentials_error
    user_id = payload.get("sub")
    if user_id is None:
        raise _credentials_error

    user = db.get(User, int(user_id))
    if user is None or not user.is_active:
        raise _credentials_error
    return user


def require_role(role: str):
    """Dependency factory enforcing a required role (e.g. 'admin')."""

    def _guard(current_user: User = Depends(get_current_user)) -> User:
        if current_user.role != role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions",
            )
        return current_user

    return _guard


def get_cache() -> Cache:
    global _cache
    if _cache is None:
        _cache = build_cache(get_config().cache)
    return _cache


def _api_key_from_request(request: Request, token: str | None) -> str | None:
    header = request.headers.get("x-api-key")
    if header:
        return header
    if token and token.startswith(KEY_PREFIX):
        return token
    return None


def get_principal(
    request: Request,
    token: str | None = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> Principal:
    """Resolve the caller: an API key (header or bearer) or a JWT user."""
    plaintext = _api_key_from_request(request, token)
    if plaintext is not None:
        key = verify_api_key(db, plaintext)
        if key is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
        if not check_rate_limit(get_cache(), f"key:{key.id}", key.rate_limit_per_minute):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Rate limit exceeded"
            )
        return principal_for_api_key(db, key)
    user = get_current_user(token=token, db=db)
    return principal_for_user(db, user)


def get_access_filter(
    principal: Principal = Depends(get_principal), db: Session = Depends(get_db)
) -> AccessFilter:
    return compute_access_filter(db, principal)


# --- Task 10: dependencies onto the real strategy, LLM and stores -------------
#
# Built once per process, the same manual singleton pattern as ``get_cache``
# above (a module-global default of ``None``, filled in on first use). Task 16
# replaces all four of these with objects on ``app.state``; this is not the
# place to fix that, only to stop wiring search onto the v1 in-memory index.

_strategy_registry: StrategyRegistry | None = None


def get_strategy_registry() -> StrategyRegistry:
    global _strategy_registry
    if _strategy_registry is None:
        from ragfabric_core.strategies.registry_defaults import default_registry

        _strategy_registry = default_registry(get_config(), get_session_factory())
    return _strategy_registry


_llm_provider: LLMProvider | None = None


def get_llm_provider() -> LLMProvider:
    global _llm_provider
    if _llm_provider is None:
        from ragfabric_core.providers.registry import build_llm_provider

        _llm_provider = build_llm_provider(get_config().llm)
    return _llm_provider


_rerankers: dict[str, Reranker] = {}


def get_reranker(kind: str, llm: LLMProvider | None = None) -> Reranker:
    """The process wide reranker instance for ``kind``, built once and reused.

    Same manual module-level singleton pattern as ``get_cache``/
    ``get_llm_provider`` above, keyed by kind rather than a single default,
    because Task 12 lets a request name any of the three kinds and each one
    needs its own cached instance. Deliberately NOT ``functools.lru_cache``:
    this package uses manual module-level globals throughout and
    ``lru_cache`` appears nowhere else in it.

    Sharing a reranker across requests is safe because rerankers carry no
    per-request mutable state: Task 8 rejected a mutable call counter on the
    reranker specifically because the registry already shares one instance
    across every concurrent request, which established that ``NoopReranker``
    and ``LlmReranker`` hold only read-only references. ``CrossEncoderReranker``
    is the case this cache actually exists for: its only mutable state is its
    lazily loaded model (loaded from disk on first use, per Task 6's ruling
    that the load must not block server startup), and that cached model is
    exactly the state a per-kind cache is meant to share. Without this cache,
    a per-request ``TraditionalRAGStrategy`` built fresh around a brand new
    ``CrossEncoderReranker()`` would reload that model from disk on every
    single request naming ``rerank: "cross_encoder"``, turning a one-word
    request body into a way to force a multi-gigabyte reload per call.
    """
    if kind not in _rerankers:
        from ragfabric_core.config_file import RerankerConfig
        from ragfabric_core.rerank.registry import build_reranker

        _rerankers[kind] = build_reranker(RerankerConfig(kind=kind), llm=llm)
    return _rerankers[kind]


_vector_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        from ragfabric_core.providers.registry import build_embedding_provider
        from ragfabric_core.stores.registry import build_vector_store

        cfg = get_config()
        embedder = build_embedding_provider(cfg.embeddings)
        _vector_store = build_vector_store(
            cfg.vector_store, get_session_factory(), embedding_model=embedder.model
        )
    return _vector_store


_lexical_store: LexicalStore | None = None


def get_lexical_store() -> LexicalStore:
    global _lexical_store
    if _lexical_store is None:
        from ragfabric_core.stores.registry import build_lexical_store

        _lexical_store = build_lexical_store(get_config().lexical_store, get_session_factory())
    return _lexical_store


def get_embedding_model(store: VectorStore = Depends(get_vector_store)) -> str:
    """The active embedding model, read off the store's own public property.

    Never a hardcoded string and never a private attribute: a later task
    (Phase 8) compares ``embedding_model`` across strategies, so it must be
    what the store actually pinned, not a guess about what it might be.
    """
    return store.model or "unknown"
