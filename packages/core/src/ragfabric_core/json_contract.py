"""Reading a JSON object out of a model response, and refusing one that does not fit.

Shared by every module that asks a model for structured output (the agent
contracts and the graph contracts). The rule is the same everywhere: forgiving
about *packaging*, strict about *content*. Small models wrap JSON in a fenced
block, preface it with a sentence of agreement, or explain themselves
afterwards, and none of that is a real failure. A response that does not
validate against the schema is, and it comes back as a ``ContractViolation``
rather than an exception, so the caller decides whether to retry or fall back.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, ValidationError

FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


class ContractViolation(BaseModel):
    """A model response that could not be honoured.

    ``raw`` is kept because a violation that does not say what the model
    actually returned cannot be diagnosed later from a trace.
    """

    contract: str
    raw: str
    error: str


def json_candidates(text: str) -> list[str]:
    """Every substring of the response that might be the JSON object.

    Tried in order: fenced code blocks first, because when a model uses one it
    is almost always the real payload, then the widest brace-balanced span.
    """
    found = [block.strip() for block in FENCE.findall(text)]
    depth = 0
    start = -1
    for index, char in enumerate(text):
        if char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                found.append(text[start : index + 1])
                start = -1
    return found


def load_json_object(text: str) -> tuple[dict[str, Any] | None, str]:
    """The first candidate that decodes to a JSON object, or ``None`` and why not."""
    for candidate in json_candidates(text):
        try:
            loaded = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(loaded, dict):
            return loaded, ""
    return None, "no JSON object found in the response"


def parse_contract(text: str, contract: str, model: type[BaseModel]) -> Any:
    """``model`` validated from the response, or a ``ContractViolation`` naming ``contract``."""
    payload, error = load_json_object(text)
    if payload is None:
        return ContractViolation(contract=contract, raw=text, error=error)
    try:
        return model.model_validate(payload)
    except ValidationError as exc:
        return ContractViolation(contract=contract, raw=text, error=str(exc))
