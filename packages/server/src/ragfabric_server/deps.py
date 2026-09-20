"""Shared FastAPI dependencies: current-user resolution and role guards.

- ``get_current_user`` decodes the bearer access token and loads the user.
- ``require_role("admin")`` builds a dependency that also checks the role.
- ``get_principal`` resolves the caller as a Principal: an API key (header or
  bearer) or a signed in user, so route handlers can depend on one thing
  regardless of how the caller authenticated.
- ``get_access_filter`` computes the AccessFilter for that principal.
- ``get_cache``, ``get_llm_provider``, ``get_strategy_registry``,
  ``get_vector_store`` and ``get_lexical_store`` read their objects off
  ``app.state``, where ``ragfabric_server.main``'s lifespan builds each one
  once at application startup (see that module).

401 Unauthorized -> we don't know who you are (bad/missing/expired token).
403 Forbidden    -> we know who you are, but you're not allowed.
429 Too Many Requests -> an API key went over its rate limit.
"""

import threading

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
from ragfabric_core.security import ACCESS, JWTError, decode_token
from ragfabric_core.stores.base import Cache, LexicalStore, VectorStore
from ragfabric_core.strategies.base import StrategyRegistry

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/auth/login", auto_error=False)


def _credentials_error() -> HTTPException:
    """A fresh 401 each time, never a shared instance.

    A single module-level ``HTTPException`` object used to be raised from
    every failed-auth branch below. FastAPI's exception handling does not
    mutate it, so sharing it was not a correctness bug today, but it is
    exactly the kind of shared mutable state this task removes on principle:
    nothing stops a future change (attaching request-specific detail to the
    exception, for instance) from turning this into one request's 401
    leaking detail into another's response.
    """
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )


def get_current_user(
    token: str | None = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:
    """Return the active user for the request's access token."""
    if not token:
        raise _credentials_error()
    try:
        payload = decode_token(token)
    except JWTError as exc:
        raise _credentials_error() from exc

    if payload.get("type") != ACCESS:
        raise _credentials_error()
    user_id = payload.get("sub")
    if user_id is None:
        raise _credentials_error()

    user = db.get(User, int(user_id))
    if user is None or not user.is_active:
        raise _credentials_error()
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


def get_cache(request: Request) -> Cache:
    """The process wide Cache, built once at application startup.

    ``ragfabric_server.main``'s lifespan builds this (and the four
    dependencies below) exactly once, on ``app.state``, rather than each
    living as a module-level global filled in on first use: a module global
    is process-wide, permanently, with no way for a test (or a second
    ``FastAPI()`` instance in the same process) to get a clean one back.
    ``app.state`` is scoped to the one ``app`` that built it, so a fresh
    ``TestClient(app)`` genuinely gets a fresh cache/provider/registry/store
    set, the same as a fresh server process would.
    """
    return request.app.state.cache


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
        if not check_rate_limit(get_cache(request), f"key:{key.id}", key.rate_limit_per_minute):
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
# Built once at application startup (``ragfabric_server.main``'s lifespan)
# and read off ``app.state`` (see ``get_cache`` above for why). Note what is
# NOT here: ``get_reranker``, further down, deliberately keeps its
# module-level cache. Unlike these four, it is called directly (with no
# ``Request`` anywhere in scope) from ``search.py``'s ``_strategy_for`` and
# from several unit tests in ``test_overrides.py`` that build a
# ``StrategyRegistry`` by hand and never construct a FastAPI ``app`` at all.
# Moving it onto ``app.state`` would mean threading a ``Request`` through
# that helper and rewriting those tests around a live app, which is a second
# structural change riding on this one; left alone, reported as such.


def get_strategy_registry(request: Request) -> StrategyRegistry:
    return request.app.state.strategy_registry


def get_llm_provider(request: Request) -> LLMProvider:
    return request.app.state.llm_provider


_rerankers: dict[str, Reranker] = {}
_rerankers_lock = threading.Lock()


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

    The check-then-set on ``_rerankers`` is not atomic by itself: two threads
    can both see ``kind not in _rerankers`` before either has finished
    building, and both go on to build. Harmless for ``none``/``llm``, which
    are cheap to construct, but for ``cross_encoder`` it means two concurrent
    first requests can each kick off a multi-gigabyte model load at once,
    exactly the cost this cache exists to prevent. The lock below serialises
    the build; the fast path (once a kind is already cached) stays lock-free,
    and the re-check after acquiring the lock is what stops a second thread,
    unblocked after the first thread finishes building, from building again.
    """
    if kind in _rerankers:
        return _rerankers[kind]

    with _rerankers_lock:
        if kind not in _rerankers:
            from ragfabric_core.config_file import RerankerConfig
            from ragfabric_core.rerank.registry import build_reranker

            _rerankers[kind] = build_reranker(RerankerConfig(kind=kind), llm=llm)
        return _rerankers[kind]


def get_vector_store(request: Request) -> VectorStore:
    return request.app.state.vector_store


def get_lexical_store(request: Request) -> LexicalStore:
    return request.app.state.lexical_store


def get_embedding_model(store: VectorStore = Depends(get_vector_store)) -> str:
    """The active embedding model, read off the store's own public property.

    Never a hardcoded string and never a private attribute: a later task
    (Phase 8) compares ``embedding_model`` across strategies, so it must be
    what the store actually pinned, not a guess about what it might be.
    """
    return store.model or "unknown"
