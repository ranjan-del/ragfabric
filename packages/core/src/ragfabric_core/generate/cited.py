"""Generate a cited answer from retrieved chunks, and verify it before returning.

The model is told to cite with [n], to quote only verbatim, and to say it cannot
find the answer rather than guess. The output is then checked against the
citation contract. A violation is retried once with the specific reason quoted
back, because a named failure is repairable and a generic "try again" is not. A
second violation falls back to the extractive generator, which satisfies the
contract by construction, and the result records which generator produced it so
nothing downstream mistakes a fallback for a model answer.

Every call made to the LLM provider (accepted or rejected) is a real cost, so
``CitedAnswer.input_tokens`` and ``.output_tokens`` are the sum over the calls
actually issued on this run, never an estimate and never inferred from the
number of attempts allowed. ``retried`` is true only when a second call was
actually issued, and ``generator`` names whichever generator produced the
returned text, including on the no-chunks path where no call is made at all.
"""

from __future__ import annotations

import time
from typing import Literal

from pydantic import BaseModel

from ragfabric_core.generate import llm as extractive
from ragfabric_core.generate.contract import (
    NO_EVIDENCE,
    CitationViolation,
    assert_citation_contract,
)
from ragfabric_core.providers.base import LLMProvider, Message
from ragfabric_core.strategies.base import RetrievedChunk

# The exact sentence the contract's NO_EVIDENCE substring must match. Defined
# from that one sentinel, not restated, so the sentence the generator emits and
# the substring the contract checks for cannot drift apart (Ruling B).
NO_EVIDENCE_ANSWER = f"I {NO_EVIDENCE} an answer to that in the documents provided."

_SYSTEM = (
    "You answer strictly from the numbered passages provided. "
    "Cite every claim with the passage number in square brackets, for example [1]. "
    "Quote verbatim only, inside double quotation marks; paraphrase everything else. "
    "If the passages do not answer the question, reply exactly: "
    f"{NO_EVIDENCE_ANSWER}"
)


class CitedAnswer(BaseModel):
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    generator: Literal["llm", "extractive"] = "llm"
    retried: bool = False


def build_prompt(query: str, chunks: list[RetrievedChunk], violation: str | None = None) -> str:
    passages = "\n\n".join(f"[{i}] {c.text}" for i, c in enumerate(chunks, start=1))
    prompt = f"Question: {query}\n\nPassages:\n{passages}"
    if violation is not None:
        prompt += (
            "\n\nYour previous answer was rejected: "
            f"{violation}. Answer again, obeying the citation rules exactly."
        )
    return prompt


def generate_cited_answer(
    query: str,
    chunks: list[RetrievedChunk],
    llm: LLMProvider,
    *,
    model: str | None = None,
    max_tokens: int = 800,
    extractive_fallback: bool = True,
) -> CitedAnswer:
    started = time.perf_counter()
    if not chunks:
        # No call is made on this path, so both token counts are exactly zero,
        # never an estimate of what a call would have cost.
        # "extractive" here means "not model-generated", the honest label for
        # a canned sentence, not a claim that the extractive generator ran:
        # extractive_answer() is never called on this path either.
        return CitedAnswer(
            text=NO_EVIDENCE_ANSWER,
            model="none",
            generator="extractive",
            latency_ms=int((time.perf_counter() - started) * 1000),
        )

    violation: str | None = None
    input_tokens = 0
    output_tokens = 0
    last_model = model or getattr(llm, "default_model", "unknown")

    for attempt in (1, 2):
        completion = llm.complete(
            [
                Message(role="system", content=_SYSTEM),
                Message(role="user", content=build_prompt(query, chunks, violation)),
            ],
            model=model,
            max_tokens=max_tokens,
        )
        # This call actually happened, whether or not its answer is accepted,
        # so its tokens are counted immediately and unconditionally.
        input_tokens += completion.input_tokens
        output_tokens += completion.output_tokens
        last_model = completion.model
        try:
            assert_citation_contract(completion.text, chunks)
        except CitationViolation as exc:
            violation = exc.reason
            if attempt == 2:
                break
            continue
        return CitedAnswer(
            text=completion.text,
            model=completion.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=int((time.perf_counter() - started) * 1000),
            generator="llm",
            # True only when this is the second attempt, i.e. a second call
            # was actually issued after the first was rejected.
            retried=attempt == 2,
        )

    if not extractive_fallback:
        raise CitationViolation(violation or "citation contract not satisfied")

    # The extractive generator makes no LLM call of its own, so it adds
    # nothing to input_tokens/output_tokens: the counts stay exactly the sum
    # of the two rejected calls that actually ran above.
    text = extractive.extractive_answer(query, [c.model_dump() for c in chunks])
    return CitedAnswer(
        text=text,
        model=last_model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=int((time.perf_counter() - started) * 1000),
        generator="extractive",
        # True because reaching the fallback required two real calls above.
        retried=True,
    )
