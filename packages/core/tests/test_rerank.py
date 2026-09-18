import pytest

from ragfabric_core.providers.base import Completion, ProviderError
from ragfabric_core.rerank.llm_reranker import LlmReranker
from ragfabric_core.rerank.noop import NoopReranker
from ragfabric_core.rerank.registry import build_reranker
from ragfabric_core.strategies.base import RetrievedChunk


def chunk(cid: int, text: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid, document_id=1, collection_id=None, text=text, score=score
    )


class StubLLM:
    name = "stub"
    default_model = "stub"

    def __init__(self, scores: list[float]) -> None:
        self._scores = scores
        self.calls = 0

    def complete(self, messages, *, model=None, max_tokens=1024, temperature=0.0, json_schema=None):
        self.calls += 1
        import json

        payload = json.dumps({"scores": self._scores})
        return Completion(
            text=payload,
            model="stub",
            provider="stub",
            input_tokens=1,
            output_tokens=1,
            latency_ms=1,
        )


def test_noop_reranker_truncates_but_does_not_reorder():
    chunks = [chunk(1, "a", 0.9), chunk(2, "b", 0.8), chunk(3, "c", 0.7)]
    out = NoopReranker().rerank("q", chunks, top_k=2)
    assert [c.chunk_id for c in out] == [1, 2]


def test_llm_reranker_reorders_by_the_returned_scores():
    chunks = [chunk(1, "a", 0.9), chunk(2, "b", 0.8), chunk(3, "c", 0.7)]
    out = LlmReranker(StubLLM([0.1, 0.95, 0.5])).rerank("q", chunks, top_k=3)
    assert [c.chunk_id for c in out] == [2, 3, 1]
    assert out[0].score == pytest.approx(0.95)


def test_llm_reranker_returns_a_subset_and_never_invents_a_chunk():
    chunks = [chunk(1, "a", 0.9), chunk(2, "b", 0.8)]
    out = LlmReranker(StubLLM([0.2, 0.4])).rerank("q", chunks, top_k=5)
    assert {c.chunk_id for c in out} <= {1, 2}
    assert len(out) == 2


def test_llm_reranker_falls_back_to_the_original_order_when_the_model_misbehaves():
    class Broken(StubLLM):
        def complete(self, messages, **kwargs):
            return Completion(
                text="not json at all",
                model="stub",
                provider="stub",
                input_tokens=1,
                output_tokens=1,
                latency_ms=1,
            )

    chunks = [chunk(1, "a", 0.9), chunk(2, "b", 0.8)]
    out = LlmReranker(Broken([])).rerank("q", chunks, top_k=2)
    assert [c.chunk_id for c in out] == [1, 2], (
        "a malformed rerank response must degrade to retrieval order, not drop results"
    )


def test_llm_reranker_handles_a_wrong_length_score_list():
    chunks = [chunk(1, "a", 0.9), chunk(2, "b", 0.8), chunk(3, "c", 0.7)]
    out = LlmReranker(StubLLM([0.5])).rerank("q", chunks, top_k=3)
    assert [c.chunk_id for c in out] == [1, 2, 3]


def test_build_reranker_none_gives_the_noop():
    from ragfabric_core.config_file import RerankerConfig

    assert build_reranker(RerankerConfig(kind="none")).name == "none"


def test_build_reranker_llm_without_a_provider_is_a_clear_error():
    from ragfabric_core.config_file import RerankerConfig

    with pytest.raises(ProviderError) as exc:
        build_reranker(RerankerConfig(kind="llm"), llm=None)
    assert "llm provider" in str(exc.value)


def test_build_reranker_cross_encoder_without_the_extra_names_the_install_command(monkeypatch):
    import ragfabric_core.rerank.cross_encoder as ce
    from ragfabric_core.config_file import RerankerConfig

    def boom():
        raise ImportError("no module named sentence_transformers")

    monkeypatch.setattr(ce, "_import_cross_encoder", boom)
    with pytest.raises(ProviderError) as exc:
        build_reranker(RerankerConfig(kind="cross_encoder"))
    assert "ragfabric[rerank]" in str(exc.value)
