"""The evaluation section of ragfabric.yaml."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from ragfabric_core.config_file import EvaluationConfig, RagFabricConfig, load_config

ROOT = Path(__file__).resolve().parents[3]


def test_defaults():
    cfg = RagFabricConfig().evaluation
    assert (cfg.collection, cfg.judge, cfg.judge_model, cfg.top_k) == (
        "ragfabric-eval",
        "auto",
        None,
        5,
    )


def test_unknown_keys_and_bad_values_are_refused():
    with pytest.raises(ValidationError):
        EvaluationConfig(jugde="llm")
    with pytest.raises(ValidationError):
        EvaluationConfig(judge="gpt")
    with pytest.raises(ValidationError):
        EvaluationConfig(top_k=0)


def test_both_example_files_carry_the_section():
    for path in (
        ROOT / "ragfabric.example.yaml",
        ROOT / "packages/cli/src/ragfabric_cli/data/ragfabric.example.yaml",
    ):
        cfg = load_config(path)
        assert cfg.evaluation == EvaluationConfig(), path
        assert "evaluation:" in path.read_text()
