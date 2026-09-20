"""ragfabric_sdk: a typed HTTP client for a RagFabric server.

Talks HTTP only and never imports ragfabric_core, ragfabric_server or
ragfabric_cli; see client.py for why.
"""

from __future__ import annotations

from ragfabric_sdk.client import Client
from ragfabric_sdk.errors import AuthError, NotFoundError, RagFabricError, RateLimitError
from ragfabric_sdk.models import (
    Answer,
    AskEvent,
    Citation,
    Document,
    Highlight,
    Run,
    SearchResult,
    Source,
    SourceDocument,
    Span,
)

__all__ = [
    "Client",
    "RagFabricError",
    "AuthError",
    "NotFoundError",
    "RateLimitError",
    "Answer",
    "AskEvent",
    "Citation",
    "Document",
    "Highlight",
    "Run",
    "SearchResult",
    "Source",
    "SourceDocument",
    "Span",
]
