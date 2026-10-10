"""Build a connector from its ragfabric.yaml entry."""

from __future__ import annotations

from ragfabric_core.config_file import (
    ConnectorConfig,
    FolderConnectorConfig,
    GoogleDriveConnectorConfig,
)
from ragfabric_core.connectors.base import Connector


def build_connector(cfg: ConnectorConfig) -> Connector:
    if isinstance(cfg, FolderConnectorConfig):
        from ragfabric_core.connectors.folder import FolderConnector

        return FolderConnector(cfg.path, recursive=cfg.recursive, settle_seconds=cfg.settle_seconds)
    if isinstance(cfg, GoogleDriveConnectorConfig):
        from ragfabric_core.connectors.google_drive import GoogleDriveConnector

        return GoogleDriveConnector.from_config(cfg)
    raise ValueError(f"unknown connector kind: {cfg!r}")
