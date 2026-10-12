"""Access leak test (design D17): forbidden collection probes for every target.

Two collections are ingested from ``benchmarks/leak_corpus``: ``leak-public`` (no grants, so
open to everyone) and ``leak-secret`` (granted to one group only). A principal outside that
group asks questions answered only in ``leak-secret``, scoped to both collections, the way a
user searching everything they can see would. Per target it counts:

- forbidden chunks retrieved: any is a leak;
- unfilled slots: empty result slots (fewer than ``top_k`` chunks returned) although the open
  collection had enough chunks to fill them. A post filter that leaks nothing can still starve the answer this way.

RagFabric's targets retrieve with the principal's computed access filter. LlamaIndex and
LangChain have no access model of their own; each runs twice: as built by default (no filter)
and with its documented metadata filter on the collection.

    uv run --with-requirements benchmarks/requirements-competitors.txt \
        python benchmarks/leak_test.py --out docs/benchmarks/results/leak-test.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

PROBES = [
    "What is the proposed purchase price in Project Halcyon?",
    "When is the target signing date for the Brightwater acquisition?",
    "How many roles will be cut from the Leeds support team?",
    "What severance will affected staff receive in the restructuring?",
    "What is the internal codename for Ledger 5?",
    "When is Ledger 5 planned to launch, and what is the price rise?",
    "What happened in security incident SI-2026-07?",
    "How long did the attacker read the staging database in August 2026?",
]
RAGFABRIC_TARGETS = ("traditional", "vectorless", "auto", "agentic")


@dataclass(frozen=True)
class ProbeResult:
    retrieved: int
    forbidden: int
    unfilled: int


def score_probe(
    *, retrieved: list[int | None], forbidden: set[int], top_k: int, allowed_available: int
) -> ProbeResult:
    """``retrieved`` is the collection id of each returned chunk, in rank order."""
    n_forbidden = sum(1 for c in retrieved if c in forbidden)
    could_fill = min(top_k, allowed_available)
    return ProbeResult(
        retrieved=len(retrieved),
        forbidden=n_forbidden,
        unfilled=max(0, could_fill - len(retrieved)),
    )


def summarise(results: list[ProbeResult]) -> dict:
    return {
        "probes": len(results),
        "probes_leaking": sum(1 for r in results if r.forbidden),
        "forbidden_chunks": sum(r.forbidden for r in results),
        "unfilled_slots": sum(r.unfilled for r in results),
        "retrieved_chunks": sum(r.retrieved for r in results),
    }


def _setup(factory):
    """Ingest both collections (idempotent) and restrict the secret one to its own group."""
    from ragfabric_core.auth.service import create_group, grant_collection
    from ragfabric_core.evaluation.corpus import collection_id, ingest_corpus
    from ragfabric_core.models.access import CollectionGrant, Group

    for name in ("public", "secret"):
        paths = sorted((HERE / "leak_corpus" / name).glob("*.txt"))
        ingest_corpus(factory, f"leak-{name}", paths)
    with factory() as db:
        public = collection_id(db, "leak-public")
        secret = collection_id(db, "leak-secret")
        group = db.query(Group).filter(Group.name == "leak-secret-team").first()
        if group is None:
            group = create_group(db, "leak-secret-team", "Phase 11 leak test")
        if not db.query(CollectionGrant).filter_by(group_id=group.id, collection_id=secret).first():
            grant_collection(db, group.id, secret, "read")
        db.commit()
    return public, secret


def run(out: Path, frameworks: bool) -> dict:
    from ragfabric_core.auth.policy import compute_access_filter
    from ragfabric_core.auth.principal import Principal
    from ragfabric_core.models.document import Chunk
    from ragfabric_core.providers.registry import answering_model, build_llm_provider
    from ragfabric_core.runtime import get_config, get_session_factory
    from ragfabric_core.strategies.base import RetrievalContext, StrategyName, StrategyParams
    from ragfabric_core.strategies.registry_defaults import default_registry

    cfg = get_config()
    factory = get_session_factory()
    public, secret = _setup(factory)
    top_k = cfg.evaluation.top_k
    probe = Principal(user_id=None, email=None, role="user", group_ids=[])
    with factory() as db:
        access = compute_access_filter(db, probe)
        allowed_available = db.query(Chunk).filter(Chunk.collection_id == public).count()
        chunk_collection = dict(db.query(Chunk.id, Chunk.collection_id))
    if access.allows(None, secret):
        raise SystemExit("set up failed: the probing principal can read the secret collection")

    report: dict = {
        "top_k": top_k,
        "probes": PROBES,
        "allowed_chunks_available": allowed_available,
        "targets": {},
    }
    registry = default_registry(cfg, factory)
    for name in RAGFABRIC_TARGETS:
        strategy = registry.get(StrategyName(name))
        results, errors = [], []
        for question in PROBES:
            ctx = RetrievalContext(
                principal=probe,
                access_filter=access,
                collection_ids=[public, secret],
                params=StrategyParams(top_k=top_k),
            )
            try:
                result = strategy.retrieve(question, ctx)
            except Exception as exc:  # recorded, never scored as zero leaks
                errors.append(f"{type(exc).__name__}: {exc}")
                continue
            cols = [c.collection_id for c in result.chunks]
            results.append(
                score_probe(
                    retrieved=cols,
                    forbidden={secret},
                    top_k=top_k,
                    allowed_available=allowed_available,
                )
            )
        report["targets"][f"ragfabric:{name}"] = {
            **summarise(results),
            "errors": errors,
            "per_probe": [asdict(r) for r in results],
        }
        print(name, report["targets"][f"ragfabric:{name}"] | {"per_probe": "..."}, flush=True)

    if frameworks:
        from adapters import langchain_target, llamaindex_target
        from adapters.common import load_chunks

        llm = build_llm_provider(cfg.llm)
        model = answering_model(cfg.llm, llm)
        base = (cfg.llm.base_url or "").removesuffix("/").removesuffix("/v1")
        chunks = load_chunks(factory, [public, secret])
        for fw, module in (("llamaindex", llamaindex_target), ("langchain", langchain_target)):
            for variant, allowed in (("default", None), ("metadata-filter", {public})):
                target = module.build(
                    chunks, model=model, top_k=top_k, base_url=base, allowed_collections=allowed
                )
                results = []
                for question in PROBES:
                    passages = target._retrieve(question)
                    cols = [
                        p.collection_id
                        if p.collection_id is not None
                        else chunk_collection.get(p.chunk_id)
                        for p in passages
                    ]
                    results.append(
                        score_probe(
                            retrieved=cols,
                            forbidden={secret},
                            top_k=top_k,
                            allowed_available=allowed_available,
                        )
                    )
                key = f"{fw}:{variant}"
                report["targets"][key] = {
                    **summarise(results),
                    "errors": [],
                    "per_probe": [asdict(r) for r in results],
                }
                print(key, report["targets"][key] | {"per_probe": "..."}, flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 11 access leak test")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--no-frameworks", action="store_true")
    args = parser.parse_args(argv)
    run(args.out, frameworks=not args.no_frameworks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
