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

import re
import time
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, Field

from ragfabric_core.generate import llm as extractive
from ragfabric_core.generate.contract import (
    NO_EVIDENCE,
    CitationViolation,
    assert_citation_contract,
)
from ragfabric_core.providers.base import LLMProvider, Message
from ragfabric_core.strategies.base import RetrievedChunk, SubQuestionReport

# The exact sentence the contract's NO_EVIDENCE substring must match. Defined
# from that one sentinel, not restated, so the sentence the generator emits and
# the substring the contract checks for cannot drift apart (Ruling B).
NO_EVIDENCE_ANSWER = f"I {NO_EVIDENCE} an answer to that in the documents provided."

# Public (not module-private) because the ask route streams a completion with
# this exact same system prompt before checking the result against the same
# citation contract, and a duplicated copy of the wording could drift from
# what generate_cited_answer actually asks the model to do.
SYSTEM_PROMPT = (
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
                Message(role="system", content=SYSTEM_PROMPT),
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


# ---------------------------------------------------------------------------
# Generation over an agent's evidence pool.
#
# Two things differ from generate_cited_answer above, and only two.
#
# **An unsupported claim is removed rather than retried.** The contract is the
# same one, unchanged, applied claim by claim instead of once over the whole
# answer. A claim that fails it is deleted and the deletion is recorded. The
# agent does not go back to retrieval for it: a claim the evidence does not
# support is usually the model saying more than it was given, and retrieving
# again does not make it true. Retrying the whole answer, which is what the
# single-shot generator does, is also wrong here: it spends a second call to
# regenerate the parts that were already fine.
#
# Applying the contract per claim is STRICTER than applying it once. Rule 2
# checks a quote against the chunks that claim cites, not against every chunk
# the answer cites anywhere, so a quote borrowed from a chunk the claim does
# not cite is caught here and is not caught by a whole-answer check. That is
# why no second whole-answer check is needed afterwards: anything that passes
# claim by claim passes as a whole.
#
# **Effective dates are surfaced, never adjudicated.** When the chunks
# answering one sub-question come from documents carrying different effective
# dates, every one of them is listed with its date. Nothing says which is
# right, or even that they disagree: deciding that is a semantic judgement, it
# is wrong often enough on a small model to be dangerous, and Phase 8 is where
# it can be measured. Absent metadata produces nothing at all, never an
# inferred date (ADR 0004).
# ---------------------------------------------------------------------------

# The chunk metadata key an effective date is read from. Nothing in the
# ingestion pipeline writes this key today, so this reports dates only for
# deployments whose store carries them, and reports nothing for the rest. That
# is the honest behaviour: a date that was never recorded cannot be produced by
# reading the ingestion timestamp instead, which measures when a file was
# uploaded and not when its contents took effect.
EFFECTIVE_DATE_KEY = "effective_date"

_MARKER_ONLY = re.compile(r"\[\d+\]")
_SENTENCE_END = frozenset(".!?")


class DroppedClaim(BaseModel):
    """One claim removed from the answer, and the contract rule that removed it."""

    text: str
    reason: str


class DatedSource(BaseModel):
    """One document that answered a sub-question, with the date it carries."""

    marker: int
    chunk_id: int
    document_id: int
    effective_date: str


class DatedSubQuestion(BaseModel):
    """The dated sources for one sub-question, when they do not all agree on a date."""

    sub_question: str
    sources: list[DatedSource]


class SubQuestionEvidence(BaseModel):
    """Which chunks answered which sub-question.

    Passed in rather than inferred, because the agent knows the grouping and
    nothing downstream can reconstruct it from a flat list of chunks.
    """

    text: str
    chunks: list[RetrievedChunk]


class AgenticAnswer(BaseModel):
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    generator: Literal["llm", "extractive"] = "llm"
    dropped_claims: list[DroppedClaim] = Field(default_factory=list)
    dated_sources: list[DatedSubQuestion] = Field(default_factory=list)


def split_claims(answer: str) -> list[str]:
    """Split an answer into claims, without cutting a quotation in half.

    A claim is a sentence, because a sentence is the unit a citation attaches
    to and the unit a reader would delete by hand. Two details matter:

    Quotation is respected. A quoted passage frequently contains its own full
    stops, and a splitter that ignored them would cut a verified quote into
    fragments, each of which then fails quote fidelity and gets dropped. The
    answer would lose the one part of it that was provably faithful.

    A fragment carrying no words of its own is joined back onto the claim
    before it. That is what a trailing marker ("... is five attempts. [1]")
    looks like after splitting, and treating it as a claim would both drop it
    as uncited prose and strip the citation off the claim it belongs to.
    """
    pieces: list[str] = []
    current: list[str] = []
    in_quote = False
    for char in answer:
        current.append(char)
        if char == '"':
            in_quote = not in_quote
        elif char in _SENTENCE_END and not in_quote:
            pieces.append("".join(current))
            current = []
    if current:
        pieces.append("".join(current))

    claims: list[str] = []
    for piece in pieces:
        if claims and not _carries_words(piece):
            claims[-1] += piece
        else:
            claims.append(piece)
    return claims


def _carries_words(piece: str) -> bool:
    return any(char.isalnum() for char in _MARKER_ONLY.sub("", piece))


def drop_unsupported_claims(
    answer: str, chunks: list[RetrievedChunk]
) -> tuple[str, list[DroppedClaim]]:
    """Return the answer with every unsupported claim removed, and what was removed.

    The contract is imported, not reimplemented: the rule that decides is the
    same one Phase 3 ships and Phase 3's tests hold.
    """
    kept: list[str] = []
    dropped: list[DroppedClaim] = []
    for claim in split_claims(answer):
        if not _carries_words(claim):
            kept.append(claim)
            continue
        try:
            assert_citation_contract(claim, chunks)
        except CitationViolation as exc:
            dropped.append(DroppedClaim(text=claim.strip(), reason=exc.reason))
            continue
        kept.append(claim)
    return "".join(kept).strip(), dropped


def dated_sub_questions(
    evidence: Sequence[SubQuestionEvidence], chunks: list[RetrievedChunk]
) -> list[DatedSubQuestion]:
    """Sub-questions whose sources do not all carry the same effective date.

    One row per document, because two chunks of one document share its date and
    listing both would report a difference where there is none. A sub-question
    whose sources carry one date, or no date at all, produces nothing.
    """
    markers = {chunk.chunk_id: index for index, chunk in enumerate(chunks, start=1)}
    reports: list[DatedSubQuestion] = []
    for group in evidence:
        seen_documents: set[int] = set()
        sources: list[DatedSource] = []
        for chunk in group.chunks:
            date = chunk.metadata.get(EFFECTIVE_DATE_KEY)
            marker = markers.get(chunk.chunk_id)
            if not isinstance(date, str) or not date.strip() or marker is None:
                continue
            if chunk.document_id in seen_documents:
                continue
            seen_documents.add(chunk.document_id)
            sources.append(
                DatedSource(
                    marker=marker,
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    effective_date=date,
                )
            )
        if len({source.effective_date for source in sources}) < 2:
            continue
        sources.sort(key=lambda source: (source.effective_date, source.document_id))
        reports.append(DatedSubQuestion(sub_question=group.text, sources=sources))
    return reports


def dated_sources_note(reports: list[DatedSubQuestion]) -> str:
    """The sentence added to the answer for each dated difference.

    It carries markers so it is cited like any other statement, and it carries
    no quotation marks: the sub-question is the caller's own words, not a quote
    from a chunk, and wrapping it in quotes would put it in front of the
    contract's quote fidelity rule as if it were one.
    """
    lines: list[str] = []
    for report in reports:
        listed = "; ".join(
            f"document {source.document_id} dated {source.effective_date} [{source.marker}]"
            for source in report.sources
        )
        asked = report.sub_question.replace('"', "").strip()
        lines.append(
            f"The sources for this part of the question ({asked}) carry different "
            f"effective dates: {listed}."
        )
    return " ".join(lines)


class AgenticContractResult(BaseModel):
    """The answer after the contract has been applied claim by claim."""

    text: str
    dropped_claims: list[DroppedClaim] = Field(default_factory=list)
    dated_sources: list[DatedSubQuestion] = Field(default_factory=list)


def apply_agentic_contract(
    text: str,
    chunks: list[RetrievedChunk],
    sub_question_evidence: Sequence[SubQuestionEvidence] = (),
) -> AgenticContractResult:
    """Drop what the contract refuses, then surface differing effective dates.

    Separate from the generation call above because the streaming route already
    has the text: it watched the model produce it token by token, and re-asking
    for an answer it has already shown the caller would spend a second call to
    rewrite the parts that were fine. Editing what is there is both cheaper and
    closer to what the caller watched arrive.

    An answer left with nothing becomes the no-evidence sentence rather than an
    empty string. Every claim having been refused is a real outcome and saying
    so is the honest form of it.
    """
    kept, dropped = drop_unsupported_claims(text, chunks)
    if not kept:
        kept = NO_EVIDENCE_ANSWER
    reports = dated_sub_questions(sub_question_evidence, chunks)
    note = dated_sources_note(reports)
    if note:
        kept = f"{kept} {note}"
    return AgenticContractResult(text=kept, dropped_claims=dropped, dated_sources=reports)


def sub_question_evidence_from(
    reports: Sequence[SubQuestionReport], chunks: list[RetrievedChunk]
) -> list[SubQuestionEvidence]:
    """Rebuild the per sub-question grouping from a result and its reports.

    The report carries chunk ids rather than chunks, because the chunks are
    already on the result and carrying them twice would double the size of
    every agentic response for no new information.
    """
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    return [
        SubQuestionEvidence(
            text=report.text,
            chunks=[by_id[chunk_id] for chunk_id in report.chunk_ids if chunk_id in by_id],
        )
        for report in reports
    ]


def generate_agentic_answer(
    query: str,
    chunks: list[RetrievedChunk],
    llm: LLMProvider,
    *,
    sub_question_evidence: Sequence[SubQuestionEvidence] = (),
    model: str | None = None,
    max_tokens: int = 800,
) -> AgenticAnswer:
    """Answer from the agent's pooled evidence, dropping what the contract refuses.

    One model call, always. There is no retry: the claims that satisfied the
    contract are kept exactly as the model wrote them, and the ones that did
    not are removed. If every claim is removed, the no-evidence sentence is
    returned rather than an empty string or an extractive substitute: the
    contract rejected everything the model said about this evidence, and
    filling the gap with different text would be answering a question the run
    just established it cannot answer from what it has.
    """
    started = time.perf_counter()
    if not chunks:
        # No call on this path, so both counts are exactly zero rather than an
        # estimate of what a call would have cost.
        return AgenticAnswer(
            text=NO_EVIDENCE_ANSWER,
            model="none",
            generator="extractive",
            latency_ms=int((time.perf_counter() - started) * 1000),
        )

    completion = llm.complete(
        [
            Message(role="system", content=SYSTEM_PROMPT),
            Message(role="user", content=build_prompt(query, chunks)),
        ],
        model=model,
        max_tokens=max_tokens,
    )
    checked = apply_agentic_contract(completion.text, chunks, sub_question_evidence)

    return AgenticAnswer(
        text=checked.text,
        model=completion.model,
        input_tokens=completion.input_tokens,
        output_tokens=completion.output_tokens,
        latency_ms=int((time.perf_counter() - started) * 1000),
        generator="llm",
        dropped_claims=checked.dropped_claims,
        dated_sources=checked.dated_sources,
    )
