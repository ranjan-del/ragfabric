"""Provider configuration: read, write and a real connection test.

The console needs to answer three questions without anyone opening a shell on
the server: which provider and model are selected, is the key that provider
needs actually present, and does a call to it work right now.

The third one is the reason this route exists at all. A configuration screen
that reports "saved" and leaves the operator to find out at query time that
the key is wrong is worse than no screen. ``POST /providers/test`` makes a
real call and reports the real outcome, including the provider's own error
text. It never reports success without a successful call.

Everything here requires admin, and every write is written to the audit log.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ragfabric_core.db.session import get_db
from ragfabric_core.models.access import AuditLog
from ragfabric_core.models.user import User
from ragfabric_core.providers.base import Message
from ragfabric_core.providers.registry import build_embedding_provider, build_llm_provider
from ragfabric_core.runtime import get_config, reset_config
from ragfabric_server import provider_config
from ragfabric_server.deps import require_role
from ragfabric_server.schemas.providers import (
    ProviderConfigOut,
    ProviderConfigUpdate,
    ProviderConfigWritten,
    ProviderTestRequest,
    ProviderTestResult,
)

router = APIRouter(dependencies=[Depends(require_role("admin"))])


@router.get("/providers", response_model=ProviderConfigOut)
def read_providers() -> ProviderConfigOut:
    """Report the selected providers and whether their secrets are present."""
    return ProviderConfigOut.model_validate(provider_config.describe(get_config()))


@router.put("/providers", response_model=ProviderConfigWritten)
def write_providers(
    payload: ProviderConfigUpdate,
    db: Session = Depends(get_db),
    admin: User = Depends(require_role("admin")),
) -> ProviderConfigWritten:
    """Persist the provider selection to ragfabric.yaml and audit the change."""
    config, changed, requires_reindex = provider_config.write_sections(
        provider_config.config_path(),
        get_config(),
        llm=payload.llm.model_dump() if payload.llm is not None else None,
        embeddings=payload.embeddings.model_dump() if payload.embeddings is not None else None,
    )
    # The process wide config cache now holds the file as it was before the
    # write, so drop it. Providers themselves are built once at startup, which
    # is what restart_required reports below.
    reset_config()

    db.add(
        AuditLog(
            principal_user_id=admin.id,
            action="provider_config_update",
            details={"changed": changed, "requires_reindex": requires_reindex},
        )
    )
    db.commit()

    return ProviderConfigWritten(
        **provider_config.describe(config),
        requires_reindex=requires_reindex,
        restart_required=bool(changed),
        changed=changed,
    )


@router.post("/providers/test", response_model=ProviderTestResult)
def test_provider(payload: ProviderTestRequest) -> ProviderTestResult:
    """Make one real call to the selected provider and report what happened.

    A broad ``except Exception`` is correct here and nowhere else in this
    package: the point of the endpoint is to turn any failure, whether a
    missing key, an unreachable host or a refused model, into a reported
    result rather than a 500. The provider's own message is passed through
    unedited, because "connection failed" tells an operator nothing.
    """
    config = get_config()
    try:
        if payload.target == "llm":
            provider = build_llm_provider(config.llm)
            completion = provider.complete([Message(role="user", content="ping")], max_tokens=1)
            return ProviderTestResult(target="llm", ok=True, detail="", model=completion.model)
        embedder = build_embedding_provider(config.embeddings)
        embedder.embed(["ping"])
        return ProviderTestResult(target="embeddings", ok=True, detail="", model=embedder.model)
    except Exception as exc:  # noqa: BLE001
        return ProviderTestResult(target=payload.target, ok=False, detail=str(exc), model=None)
