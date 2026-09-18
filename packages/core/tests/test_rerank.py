import re

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
        self.last_messages: list | None = None

    def complete(self, messages, *, model=None, max_tokens=1024, temperature=0.0, json_schema=None):
        self.calls += 1
        self.last_messages = messages
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


# --- Fix round 1: NaN / infinity must degrade like any other malformed score ---


def test_llm_reranker_treats_a_nan_score_as_malformed():
    chunks = [chunk(1, "a", 0.9), chunk(2, "b", 0.8), chunk(3, "c", 0.7)]
    out = LlmReranker(StubLLM([float("nan"), 0.5, 0.9])).rerank("q", chunks, top_k=3)
    assert [c.chunk_id for c in out] == [1, 2, 3], (
        "a NaN score must degrade the whole rerank to retrieval order, not a partial rescore"
    )


def test_llm_reranker_treats_an_infinite_score_as_malformed():
    chunks = [chunk(1, "a", 0.9), chunk(2, "b", 0.8), chunk(3, "c", 0.7)]
    out = LlmReranker(StubLLM([float("inf"), 0.5, 0.9])).rerank("q", chunks, top_k=3)
    assert [c.chunk_id for c in out] == [1, 2, 3]


# --- Fix round 1: sort on the raw score, clamp only the stored value ---


def test_llm_reranker_sorts_by_raw_score_but_clamps_the_stored_score():
    chunks = [chunk(1, "a", 0.9), chunk(2, "b", 0.8), chunk(3, "c", 0.7)]
    out = LlmReranker(StubLLM([5.0, -2.0, 0.5])).rerank("q", chunks, top_k=3)
    assert [c.chunk_id for c in out] == [1, 3, 2], (
        "ordering must follow the raw scores (5.0 > 0.5 > -2.0), not a clamped view of them"
    )
    assert all(0.0 <= c.score <= 1.0 for c in out)
    assert out[0].score == pytest.approx(1.0)
    assert out[-1].score == pytest.approx(0.0)


# --- Fix round 1: the reranker owns its own prompt budget ---


def test_llm_reranker_caps_candidates_sent_to_the_model():
    chunks = [chunk(i, f"text-{i}", 1.0) for i in range(1, 21)]
    stub = StubLLM([0.5] * 5)
    out = LlmReranker(stub, max_candidates=5).rerank("q", chunks, top_k=3)
    assert stub.calls == 1
    prompt = stub.last_messages[1].content
    assert len(re.findall(r"\[\d+\]", prompt)) == 5
    assert "text-1" in prompt
    assert "text-6" not in prompt
    assert len(out) == 3


def test_llm_reranker_effective_cap_rises_to_match_top_k():
    chunks = [chunk(i, f"text-{i}", 1.0) for i in range(1, 21)]
    stub = StubLLM([0.5] * 8)
    LlmReranker(stub, max_candidates=5).rerank("q", chunks, top_k=8)
    prompt = stub.last_messages[1].content
    assert len(re.findall(r"\[\d+\]", prompt)) == 8
    assert "text-8" in prompt
    assert "text-9" not in prompt


def test_llm_reranker_default_cap_does_not_reduce_result_count_below_top_k():
    chunks = [chunk(i, f"text-{i}", 1.0) for i in range(1, 6)]
    stub = StubLLM([0.1, 0.2, 0.3, 0.4, 0.5])
    out = LlmReranker(stub).rerank("q", chunks, top_k=5)
    assert len(out) == 5


# --- Fix round 1: the cross encoder loads its model lazily, but validates the extra eagerly ---


def test_cross_encoder_reranker_loads_the_model_lazily_and_caches_it(monkeypatch):
    import ragfabric_core.rerank.cross_encoder as ce

    class SentinelModel:
        instances = 0

        def __init__(self, model_name: str) -> None:
            SentinelModel.instances += 1
            self.model_name = model_name

        def predict(self, pairs):
            return [0.5 for _ in pairs]

    monkeypatch.setattr(ce, "_import_cross_encoder", lambda: SentinelModel)

    reranker = ce.CrossEncoderReranker()
    assert SentinelModel.instances == 0, "construction must validate the extra without loading weights"

    chunks = [chunk(1, "a", 0.9), chunk(2, "b", 0.8)]
    reranker.rerank("q", chunks, top_k=2)
    assert SentinelModel.instances == 1

    reranker.rerank("q", chunks, top_k=2)
    assert SentinelModel.instances == 1, "a second call must reuse the cached model, not reload it"


def test_cross_encoder_reranker_degrades_to_retrieval_order_on_a_nan_score(monkeypatch):
    import ragfabric_core.rerank.cross_encoder as ce

    class NanModel:
        def __init__(self, model_name: str) -> None:
            pass

        def predict(self, pairs):
            return [float("nan"), 0.5, 0.9]

    monkeypatch.setattr(ce, "_import_cross_encoder", lambda: NanModel)

    reranker = ce.CrossEncoderReranker()
    chunks = [chunk(1, "a", 0.9), chunk(2, "b", 0.8), chunk(3, "c", 0.7)]
    out = reranker.rerank("q", chunks, top_k=3)
    assert [c.chunk_id for c in out] == [1, 2, 3]


def test_cross_encoder_reranker_sorts_by_raw_score_but_clamps_the_stored_score(monkeypatch):
    import ragfabric_core.rerank.cross_encoder as ce

    class WideRangeModel:
        def __init__(self, model_name: str) -> None:
            pass

        def predict(self, pairs):
            return [5.0, -2.0, 0.5]

    monkeypatch.setattr(ce, "_import_cross_encoder", lambda: WideRangeModel)

    reranker = ce.CrossEncoderReranker()
    chunks = [chunk(1, "a", 0.9), chunk(2, "b", 0.8), chunk(3, "c", 0.7)]
    out = reranker.rerank("q", chunks, top_k=3)
    assert [c.chunk_id for c in out] == [1, 3, 2]
    assert all(0.0 <= c.score <= 1.0 for c in out)
