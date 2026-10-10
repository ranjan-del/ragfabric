"""The OpenAPI specification the TypeScript SDK is generated from (Phase 9, Task 1).

``packages/sdk-typescript/openapi.json`` is checked in, and the SDK's types are
generated from it (decision D2 of the Phase 9 design). This test is what keeps
the snapshot honest: a change to any route or schema that is not followed by a
regeneration fails here, in a CI job that already blocks a merge, rather than
surfacing later as a TypeScript client that disagrees with the server.
"""

from __future__ import annotations

import json
from pathlib import Path

from ragfabric_server.openapi import export_spec, spec_text

SNAPSHOT = Path(__file__).resolve().parents[3] / "packages" / "sdk-typescript" / "openapi.json"
REGENERATE = (
    "uv run python -m ragfabric_server.openapi packages/sdk-typescript/openapi.json "
    "&& (cd packages/sdk-typescript && npm run generate)"
)


def test_the_export_is_byte_stable_and_sorted():
    first = spec_text()
    second = spec_text()

    assert first == second
    assert first.endswith("\n")
    spec = json.loads(first)
    assert list(spec) == sorted(spec)
    assert list(spec["paths"]) == sorted(spec["paths"])


def test_ask_documents_both_of_its_response_types():
    """POST /api/ask answers JSON or an event stream; the specification says both."""
    content = export_spec()["paths"]["/api/ask"]["post"]["responses"]["200"]["content"]

    assert content["application/json"]["schema"] == {"$ref": "#/components/schemas/AnswerResponse"}
    assert "text/event-stream" in content


def test_the_run_schema_carries_the_router_fields():
    run = export_spec()["components"]["schemas"]["RunOut"]["properties"]

    assert "router_confidence" in run
    assert "router_reasoning" in run


def test_the_checked_in_snapshot_matches_the_server():
    assert SNAPSHOT.exists(), f"{SNAPSHOT} is missing. Regenerate with: {REGENERATE}"
    assert SNAPSHOT.read_text(encoding="utf-8") == spec_text(), (
        "The server's OpenAPI specification no longer matches the SDK's snapshot. "
        f"Regenerate with: {REGENERATE}"
    )
