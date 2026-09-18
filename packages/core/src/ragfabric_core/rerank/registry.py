"""Build a reranker from configuration. The only place reranker kinds are switched on."""

from __future__ import annotations

from ragfabric_core.config_file import RerankerConfig
from ragfabric_core.providers.base import LLMProvider, ProviderError
from ragfabric_core.rerank.base import Reranker
from ragfabric_core.rerank.cross_encoder import CrossEncoderReranker
from ragfabric_core.rerank.llm_reranker import LlmReranker
from ragfabric_core.rerank.noop import NoopReranker


def build_reranker(cfg: RerankerConfig, llm: LLMProvider | None = None) -> Reranker:
    if cfg.kind == "llm":
        if llm is None:
            raise ProviderError("reranker", "reranker.kind is llm but no llm provider was given")
        return LlmReranker(llm)
    if cfg.kind == "cross_encoder":
        return CrossEncoderReranker()
    return NoopReranker()
