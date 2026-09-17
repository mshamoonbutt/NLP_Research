"""Tests for the Exp 6 chosen-refusal generator's non-circularity guard."""
from __future__ import annotations

import pytest

from csjail.chosen_gen import GeneratorConfig, assert_distinct_from_judge, is_clean_refusal
from csjail.judge import JudgeConfig


def _judge_cfg(provider="openai", model="gpt-4o-mini") -> JudgeConfig:
    return JudgeConfig(
        provider=provider, model=model, temperature=0.0, max_tokens=200,
        timeout_s=30, concurrency=16, max_retries=5, rubric_prompt="x",
    )


def test_distinct_provider_and_model_passes():
    gen = GeneratorConfig(provider="anthropic", model="claude-sonnet-4-5")
    assert_distinct_from_judge(gen, _judge_cfg())  # must not raise


def test_same_provider_and_model_raises():
    gen = GeneratorConfig(provider="openai", model="gpt-4o-mini")
    with pytest.raises(ValueError):
        assert_distinct_from_judge(gen, _judge_cfg())


def test_same_provider_different_model_is_allowed():
    # Same provider but a genuinely different model is not the circularity
    # this guard exists for; only exact (provider, model) match is rejected.
    gen = GeneratorConfig(provider="openai", model="gpt-4o")
    assert_distinct_from_judge(gen, _judge_cfg())  # must not raise


def test_is_clean_refusal():
    assert is_clean_refusal(0) is True
    assert is_clean_refusal(1) is False
    assert is_clean_refusal(2) is False
    assert is_clean_refusal(None) is False
