"""Assemble one evaluation batch from configuration: collection, targets, judge.

The CLI and tests go through ``prepare`` so every ``evaluation.*`` setting is
read in one place, and a test can prove each one is.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from ragfabric_core.config_file import RagFabricConfig
from ragfabric_core.evaluation.corpus import collection_id as find_collection
from ragfabric_core.evaluation.judge import Judge, build_judge
from ragfabric_core.evaluation.strategy_target import StrategyTarget, build_targets
from ragfabric_core.pricing import PricingTable
from ragfabric_core.providers.base import LLMProvider
from ragfabric_core.providers.registry import build_llm_provider
from ragfabric_core.strategies.registry_defaults import default_registry


@dataclass
class Prepared:
    collection: str
    collection_id: int
    targets: list[StrategyTarget]
    skipped: list[tuple[str, str]]
    judge: Judge
    llm: LLMProvider
    llm_model: str | None
    embedding_model: str | None
    meta: dict = field(default_factory=dict)


def judge_model(cfg: RagFabricConfig) -> str | None:
    """``evaluation.judge_model``, else ``llm.model`` when set, else the provider default."""
    if cfg.evaluation.judge_model:
        return cfg.evaluation.judge_model
    if "model" in cfg.llm.model_fields_set:
        return cfg.llm.model
    return None


def prepare(
    cfg: RagFabricConfig,
    session_factory: Callable[[], Session],
    *,
    specs: list[str],
    collection: str | None = None,
    judge: str | None = None,
    llm: LLMProvider | None = None,
) -> Prepared:
    name = collection or cfg.evaluation.collection
    with session_factory() as db:
        cid = find_collection(db, name)
    if cid is None:
        raise ValueError(
            f"no collection named {name!r}: run ragfabric eval corpus to load the shipped corpus, "
            "or pass --collection with your own"
        )
    llm = llm or build_llm_provider(cfg.llm)
    targets, skipped = build_targets(
        specs,
        cfg,
        default_registry(cfg, session_factory),
        llm,
        collection_id=cid,
        session_factory=session_factory,
        pricing=PricingTable.load(),
    )
    chosen = build_judge(judge or cfg.evaluation.judge, llm, judge_model(cfg))
    if chosen.kind == "llm" and chosen.model is None:
        chosen.model = llm.default_model
    llm_model = cfg.llm.model if "model" in cfg.llm.model_fields_set else llm.default_model
    return Prepared(
        collection=name,
        collection_id=cid,
        targets=targets,
        skipped=skipped,
        judge=chosen,
        llm=llm,
        llm_model=llm_model,
        embedding_model=cfg.embeddings.model,
        meta={"top_k": cfg.evaluation.top_k, "llm_provider": cfg.llm.provider},
    )
