"""connectors: in ragfabric.yaml (Phase 10, Task 5)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ragfabric_core.config_file import RagFabricConfig


def _cfg(connectors):
    return RagFabricConfig.model_validate({"connectors": connectors})


def test_no_connectors_by_default():
    assert RagFabricConfig().connectors == []


def test_a_folder_connector_with_defaults():
    [c] = _cfg([{"name": "inbox", "kind": "folder", "path": "/data/inbox"}]).connectors
    assert (c.interval_seconds, c.settle_seconds, c.recursive, c.on_delete) == (
        30,
        5,
        True,
        "delete",
    )
    assert c.collection is None and c.owner is None


def test_a_drive_connector_needs_folder_and_credentials():
    [c] = _cfg(
        [
            {
                "name": "drive",
                "kind": "google_drive",
                "folder_id": "abc",
                "credentials_file": "/run/secrets/sa.json",
            }
        ]
    ).connectors
    assert c.interval_seconds == 300
    with pytest.raises(ValidationError):
        _cfg([{"name": "drive", "kind": "google_drive", "folder_id": "abc"}])


def test_duplicate_names_are_refused():
    with pytest.raises(ValidationError, match="unique"):
        _cfg(
            [
                {"name": "inbox", "kind": "folder", "path": "/a"},
                {"name": "inbox", "kind": "folder", "path": "/b"},
            ]
        )


@pytest.mark.parametrize(
    "bad",
    [
        {"name": "x", "kind": "sharepoint", "path": "/a"},
        {"name": "x", "kind": "folder"},
        {"name": "Has Space", "kind": "folder", "path": "/a"},
        {"name": "x", "kind": "folder", "path": "/a", "on_delete": "archive"},
        {"name": "x", "kind": "folder", "path": "/a", "typo": 1},
    ],
)
def test_invalid_connector_entries_are_refused(bad):
    with pytest.raises(ValidationError):
        _cfg([bad])
