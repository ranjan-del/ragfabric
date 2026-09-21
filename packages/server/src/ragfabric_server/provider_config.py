"""Read and persist the provider sections of ragfabric.yaml.

Kept here rather than in ``ragfabric_core.config_file`` on purpose. That
module is the schema and the loader, and it is deliberately read only: the
CLI and every library caller only ever load configuration. Writing it is a
server concern, driven by one admin endpoint, so the writer lives beside the
endpoint that owns it.

Two things this module refuses to do:

- It never reads, writes or echoes an API key. ``key_env_var`` names the
  environment variable a provider reads, and ``has_key`` is a boolean drawn
  from whether that variable is set. There is no code path here that can put
  a secret in a response body or in a file.
- It never rewrites a section it was not asked to change. The file is loaded
  as plain data, the named sections are replaced, and everything else is
  written back as it was found.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from ragfabric_core.config_file import (
    DEFAULT_FILENAME,
    EmbeddingsConfig,
    LLMConfig,
    RagFabricConfig,
    resolve_config_path,
)

# Which environment variable each provider reads its key from. A provider
# absent from this map needs no key: ollama talks to a local server and the
# offline doubles run in process.
KEY_ENV_VARS: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


def config_path() -> Path:
    """Where provider configuration is read from and written to.

    ``resolve_config_path`` returns None when neither RAGFABRIC_CONFIG nor a
    ragfabric.yaml in the working directory exists. A write then has to land
    somewhere, and the working directory is the same place ``ragfabric init``
    puts it, so the file is created there rather than the write failing.
    """
    resolved = resolve_config_path()
    return resolved if resolved is not None else Path.cwd() / DEFAULT_FILENAME


def status_for(section: LLMConfig | EmbeddingsConfig) -> dict[str, Any]:
    """Describe one provider section without disclosing its secret."""
    env_var = KEY_ENV_VARS.get(section.provider)
    return {
        "provider": section.provider,
        "model": section.model,
        "base_url": section.base_url,
        "dim": getattr(section, "dim", None),
        "key_env_var": env_var,
        "requires_key": env_var is not None,
        # bool() of the value, never its length and never a masked form of
        # it. A mask that preserves length leaks length.
        "has_key": bool(os.environ.get(env_var)) if env_var else False,
    }


def describe(config: RagFabricConfig) -> dict[str, Any]:
    return {"llm": status_for(config.llm), "embeddings": status_for(config.embeddings)}


def _read_raw(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _changes(prefix: str, before: dict[str, Any], after: dict[str, Any]) -> list[str]:
    keys = sorted(set(before) | set(after))
    return [f"{prefix}.{key}" for key in keys if before.get(key) != after.get(key)]


def write_sections(
    path: Path,
    current: RagFabricConfig,
    *,
    llm: dict[str, Any] | None = None,
    embeddings: dict[str, Any] | None = None,
) -> tuple[RagFabricConfig, list[str], bool]:
    """Persist the named sections and return the reloaded config.

    Returns the validated configuration, the list of changed keys, and
    whether the change invalidates the index. Validation happens against the
    whole file before anything is written, so a bad value cannot leave a file
    on disk that the application will then refuse to start against.

    "Changed" is computed against the EFFECTIVE configuration, not the raw
    file. Writing a key that was previously inherited from a default, at the
    value that default already had, changes the file but changes nothing
    about how the system behaves, and reporting it as a change would make
    every save look like it invalidated the index.

    Comments in the file are not preserved: the YAML is round tripped as
    data. That is a real cost of editing configuration from a UI, and it is
    stated here rather than discovered later.
    """
    raw = _read_raw(path)

    if llm is not None:
        raw["llm"] = llm
    if embeddings is not None:
        raw["embeddings"] = embeddings

    # Validate the whole document, not just the section that moved, so a
    # write cannot produce a file the loader would reject on next boot.
    config = RagFabricConfig.model_validate(raw)

    changed = []
    if llm is not None:
        changed += _changes("llm", current.llm.model_dump(), config.llm.model_dump())
    if embeddings is not None:
        changed += _changes(
            "embeddings", current.embeddings.model_dump(), config.embeddings.model_dump()
        )

    # ADR 0006 pins the embedding dimension of an index. Provider, model and
    # dimension each change the vectors, so any of them means every stored
    # vector was produced by something else.
    requires_reindex = any(
        key in {"embeddings.provider", "embeddings.model", "embeddings.dim"} for key in changed
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return config, changed, requires_reindex
