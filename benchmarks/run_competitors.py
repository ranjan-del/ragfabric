"""Run LlamaIndex or LangChain through the Phase 8 harness, into the same store and batch.

    uv run --with-requirements benchmarks/requirements-competitors.txt \
        python benchmarks/run_competitors.py --framework llamaindex \
        --questions <questions.json> --collection <name> --batch <batch> [--retrieval-only]

Reads RAGFABRIC_CONFIG and DATABASE_URL like the CLI (source benchmarks/env.sh). The judge is
the configured one, exactly as for RagFabric's own targets. Resumable: it always resumes.
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from adapters import langchain_target, llamaindex_target  # noqa: E402
from adapters.common import load_chunks  # noqa: E402

PACKAGES = {
    "llamaindex": ["llama-index-core", "llama-index-llms-ollama", "llama-index-embeddings-ollama"],
    "langchain": ["langchain-core", "langchain-ollama"],
}
BUILDERS = {"llamaindex": llamaindex_target.build, "langchain": langchain_target.build}


def _commit() -> str:
    import os
    import subprocess

    pinned = os.environ.get("RAGFABRIC_EVAL_COMMIT", "").strip()
    if pinned:
        return pinned
    out = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
        cwd=Path(__file__).parent,
    )
    return out.stdout.strip() if out.returncode == 0 else ""


def versions(framework: str) -> dict[str, str]:
    out = {}
    for name in PACKAGES[framework]:
        try:
            out[name] = md.version(name)
        except md.PackageNotFoundError:
            out[name] = "not installed"
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--framework", choices=sorted(BUILDERS), required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--batch", required=True)
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--judge-metric", action="append")
    args = parser.parse_args(argv)

    from ragfabric_core.evaluation.corpus import collection_id
    from ragfabric_core.evaluation.dataset import content_hash, load_questions
    from ragfabric_core.evaluation.judge import build_judge
    from ragfabric_core.evaluation.runner import run_batch
    from ragfabric_core.evaluation.session import judge_model
    from ragfabric_core.providers.registry import answering_model, build_llm_provider
    from ragfabric_core.runtime import get_config, get_session_factory

    cfg = get_config()
    factory = get_session_factory()
    with factory() as db:
        cid = collection_id(db, args.collection)
    if cid is None:
        print(f"no collection {args.collection!r}", file=sys.stderr)
        return 1
    qs = load_questions(args.questions)
    llm = build_llm_provider(cfg.llm)
    model = answering_model(cfg.llm, llm)
    base = (cfg.llm.base_url or "http://localhost:11434/v1").removesuffix("/").removesuffix("/v1")
    chunks = load_chunks(factory, [cid])
    target = BUILDERS[args.framework](
        chunks,
        model=model,
        top_k=cfg.evaluation.top_k,
        base_url=base,
        num_ctx=cfg.llm.num_ctx or 8192,
    )
    judge = build_judge(
        cfg.evaluation.judge,
        llm,
        judge_model(cfg),
        tuple(args.judge_metric) if args.judge_metric else None,
    )
    meta = {
        "commit": _commit(),
        "framework": args.framework,
        "framework_versions": versions(args.framework),
        "collection": args.collection,
        "chunks_indexed": len(chunks),
        "questions_hash": content_hash([args.questions]),
        "top_k": cfg.evaluation.top_k,
        "llm_provider": cfg.llm.provider,
        "num_ctx": cfg.llm.num_ctx,
    }
    print(
        f"batch {args.batch}: {len(qs.questions)} questions, {args.framework} "
        f"{meta['framework_versions']}, {len(chunks)} chunks, model {model}"
    )

    def progress(name, row):
        err = row.details.get("error")
        print(
            f"  {name} {row.question_id}: "
            + (
                f"error {err}"
                if err
                else f"hit {row.hit} correctness {row.correctness} {row.latency_ms}ms"
            ),
            flush=True,
        )

    ids = run_batch(
        [target],
        qs,
        judge,
        factory,
        batch=args.batch,
        meta=meta,
        llm_model=model,
        embedding_model=cfg.embeddings.model,
        on_result=progress,
        resume=True,
        retrieval_only=args.retrieval_only,
    )
    print(f"stored as run {ids[0]} (batch {args.batch})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
