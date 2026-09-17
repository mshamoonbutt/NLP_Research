"""Tests for the Exp 4b comprehension-control scoring logic."""
from __future__ import annotations

from csjail.comprehension import (
    build_paraphrase_probe,
    conditioned_asr,
    parse_comprehension,
)


def test_parse_comprehension_strict_json():
    r = parse_comprehension('{"understood": true}')
    assert r.understood is True


def test_parse_comprehension_false():
    r = parse_comprehension('{"understood": false}')
    assert r.understood is False


def test_parse_comprehension_markdown_fence():
    r = parse_comprehension('```json\n{"understood": true}\n```')
    assert r.understood is True


def test_parse_comprehension_empty_is_none():
    assert parse_comprehension("").understood is None


def test_parse_comprehension_unparseable_is_none():
    assert parse_comprehension("I cannot determine this.").understood is None


def test_build_paraphrase_probe_contains_original():
    probe = build_paraphrase_probe("mujhe kuch batao")
    assert "mujhe kuch batao" in probe


def test_conditioned_asr_buckets():
    labels =      [1, 0, 1, 0, 2, 1]
    understood =  [True, True, False, False, None, True]
    out = conditioned_asr(labels, understood)
    assert out["understood"] == {"n": 3, "asr": 2 / 3}
    assert out["not_understood"] == {"n": 2, "asr": 0.5}
    assert out["comprehension_unclear"] == {"n": 1, "asr": 0.0}
    assert out["comprehension_rate"] == 3 / 5


def test_conditioned_asr_empty_bucket_reports_none():
    out = conditioned_asr([1, 0], [True, True])
    assert out["not_understood"] == {"n": 0, "asr": None}
    assert out["comprehension_rate"] == 1.0
