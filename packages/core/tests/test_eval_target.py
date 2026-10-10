"""RagFabric strategies as evaluation targets, end to end on SQLite with the offline provider."""

import pytest

from ragfabric_core.config_file import load_config
from ragfabric_core.evaluation.corpus import collection_id, ingest_corpus
from ragfabric_core.evaluation.strategy_target import (
    build_targets,
    parse_target_spec,
)
from ragfabric_core.evaluation.target import EvalTarget
from ragfabric_core.providers.registry import build_llm_provider
from ragfabric_core.strategies.registry_defaults import default_registry


@pytest.fixture()
def loaded(eval_db, tmp_path):
    cfg_path, factory = eval_db
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "leave.md").write_text("# Leave\n\nEmployees receive 10 days of casual leave per year.")
    (docs / "travel.md").write_text("# Travel\n\nThe hotel limit is 180 US dollars per night.")
    (docs / "other.md").write_text("# Other\n\nCasual leave for contractors is 4 days.")
    ingested, skipped, failed = ingest_corpus(
        factory, "eval", [docs / "leave.md", docs / "travel.md"]
    )
    assert (sorted(ingested), skipped, failed) == (["leave.md", "travel.md"], [], [])
    ingest_corpus(factory, "elsewhere", [docs / "other.md"])
    cfg = load_config(cfg_path)
    with factory() as db:
        cid = collection_id(db, "eval")
    return cfg, factory, cid


def _targets(cfg, factory, cid, specs):
    llm = build_llm_provider(cfg.llm)
    return build_targets(
        specs,
        cfg,
        default_registry(cfg, factory),
        llm,
        collection_id=cid,
        session_factory=factory,
    )


def test_a_rerun_of_the_corpus_ingests_nothing(eval_db, tmp_path):
    _, factory = eval_db
    doc = tmp_path / "a.md"
    doc.write_text("# A\n\nSome text about leave.")
    assert ingest_corpus(factory, "c", [doc])[0] == ["a.md"]
    assert ingest_corpus(factory, "c", [doc]) == ([], ["a.md"], [])


def test_traditional_answers_with_named_ranked_contexts(loaded):
    cfg, factory, cid = loaded
    (target,), skipped = _targets(cfg, factory, cid, ["traditional"])
    assert skipped == [] and isinstance(target, EvalTarget)
    out = target.answer("How many days of casual leave?")
    assert out.answer
    assert [c.rank for c in out.contexts] == list(range(1, len(out.contexts) + 1))
    # Scoped to the evaluation collection: other.md lives elsewhere.
    assert {c.document for c in out.contexts} <= {"leave.md", "travel.md"}
    assert out.contexts[0].document == "leave.md"
    assert out.strategy_used == "traditional"
    assert out.latency_ms >= out.retrieval_latency_ms >= 0
    assert out.retrieval_calls >= 1 and out.llm_calls == 0


def test_vectorless_makes_no_embedding_call(loaded):
    cfg, factory, cid = loaded
    (target,), _ = _targets(cfg, factory, cid, ["vectorless"])
    out = target.answer("hotel limit per night")
    assert out.embedding_calls == 0 and out.contexts[0].document == "travel.md"


def test_offline_skips_agentic_graph_and_the_llm_reranker_with_reasons(loaded):
    cfg, factory, cid = loaded
    targets, skipped = _targets(
        cfg, factory, cid, ["agentic", "graph", "traditional+rerank=llm", "auto"]
    )
    assert [t.name for t in targets] == ["auto"]
    reasons = dict(skipped)
    assert "offline" in reasons["agentic"] and "offline" in reasons["graph"]
    assert "offline" in reasons["traditional+rerank=llm"]


def test_graph_disabled_is_skipped_with_the_setting(loaded):
    cfg, factory, cid = loaded
    cfg = cfg.model_copy(update={"llm": cfg.llm.model_copy(update={"provider": "ollama"})})
    _, skipped = _targets(cfg, factory, cid, ["graph"])
    assert "graph_store.enabled" in dict(skipped)["graph"]


def test_cross_encoder_without_its_extra_is_skipped(loaded, monkeypatch):
    import ragfabric_core.rerank.cross_encoder as ce

    def missing():
        raise ImportError("sentence-transformers is not installed")

    monkeypatch.setattr(ce, "_import_cross_encoder", missing)
    cfg, factory, cid = loaded
    targets, skipped = _targets(cfg, factory, cid, ["traditional+rerank=cross_encoder"])
    assert targets == [] and "not available" in dict(skipped)["traditional+rerank=cross_encoder"]


def test_rerank_none_is_a_runnable_variant(loaded):
    cfg, factory, cid = loaded
    (target,), _ = _targets(cfg, factory, cid, ["traditional+rerank=none"])
    assert target.name == "traditional+rerank=none"
    assert target.answer("casual leave").contexts


@pytest.mark.parametrize(
    "bad", ["nope", "vectorless+rerank=llm", "traditional+rerank=magic", "traditional+k=1"]
)
def test_an_unknown_spec_is_refused_naming_it(bad):
    with pytest.raises(ValueError, match="unknown target"):
        parse_target_spec(bad)


def test_every_evaluation_setting_is_read(loaded, monkeypatch):
    """Phase 5 lesson: each EvaluationConfig field must change behaviour."""
    from ragfabric_core.config_file import EvaluationConfig
    from ragfabric_core.evaluation import session as eval_session

    cfg, factory, cid = loaded
    seen = set()
    # top_k reaches the strategy's params.
    cfg2 = cfg.model_copy(update={"evaluation": EvaluationConfig(top_k=1)})
    (target,), _ = _targets(cfg2, factory, cid, ["traditional"])
    assert len(target.answer("casual leave").contexts) == 1
    seen.add("top_k")
    # collection, judge and judge_model reach the session builder.
    built = eval_session.prepare(
        cfg.model_copy(
            update={
                "evaluation": EvaluationConfig(
                    collection="eval", judge="lexical", judge_model="judge-x"
                )
            }
        ),
        factory,
        specs=["traditional"],
    )
    assert built.collection_id == cid
    seen.add("collection")
    assert built.judge.kind == "lexical"
    seen.add("judge")
    llm_cfg = cfg.model_copy(
        update={
            "llm": cfg.llm.model_copy(update={"provider": "ollama"}),
            "evaluation": EvaluationConfig(collection="eval", judge="llm", judge_model="judge-x"),
        }
    )
    assert eval_session.prepare(llm_cfg, factory, specs=["traditional"]).judge.model == "judge-x"
    seen.add("judge_model")
    assert seen == set(EvaluationConfig.model_fields)


def test_judge_model_null_means_the_llm_model(loaded):
    from ragfabric_core.config_file import EvaluationConfig
    from ragfabric_core.evaluation import session as eval_session

    cfg, factory, _ = loaded
    llm_cfg = cfg.model_copy(
        update={
            "llm": cfg.llm.model_copy(update={"provider": "ollama", "model": "llama3.1:8b"}),
            "evaluation": EvaluationConfig(collection="eval", judge="llm"),
        }
    )
    assert eval_session.prepare(llm_cfg, factory, specs=[]).judge.model == "llama3.1:8b"


def test_a_missing_collection_says_how_to_load_it(eval_db):
    from ragfabric_core.evaluation import session as eval_session

    cfg_path, factory = eval_db
    with pytest.raises(ValueError, match="ragfabric eval corpus"):
        eval_session.prepare(load_config(cfg_path), factory, specs=["traditional"])
