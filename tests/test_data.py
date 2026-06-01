"""Schema + structural tests for csjail.data."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from csjail.data import (
    DatasetError,
    Prompt,
    filter_prompts,
    load_dataset,
    pairing_coverage,
    summarize,
)
from csjail.utils.io import write_jsonl

FIXTURE = Path(__file__).parent.parent / "data" / "csjail_fixture.jsonl"


def test_fixture_loads():
    rows = load_dataset(FIXTURE)
    assert len(rows) >= 15
    assert all(isinstance(r, Prompt) for r in rows)


def test_fixture_summary_shape():
    rows = load_dataset(FIXTURE)
    s = summarize(rows)
    assert s["n_total"] == len(rows)
    assert set(s["by_condition"]).issubset({"EN", "UR", "CS", "SM"})
    assert set(s["by_harm_category"]).issubset({"H1", "H2", "H3", "H4", "H5"})
    assert s["n_base_ids_with_EN_UR_CS"] >= 5


def test_filter_by_condition():
    rows = load_dataset(FIXTURE)
    en = filter_prompts(rows, condition="EN")
    assert len(en) >= 5
    assert {r.condition for r in en} == {"EN"}


def test_filter_by_harm_category():
    rows = load_dataset(FIXTURE)
    h1 = filter_prompts(rows, harm_category="H1")
    assert all(r.harm_category == "H1" for r in h1)


def test_filter_min_authenticity():
    rows = load_dataset(FIXTURE)
    strict = filter_prompts(rows, condition="CS", min_cs_authenticity=3)
    assert all(
        (r.cs_authenticity or 0) >= 3 for r in strict if r.condition == "CS"
    )


def test_cs_row_without_style_rejected(tmp_path):
    bad = tmp_path / "bad.jsonl"
    write_jsonl(
        bad,
        [
            {
                "id": "X-001-CS",
                "base_id": "X-001",
                "harm_category": "H1",
                "condition": "CS",
                "prompt": "hello",
                "harm_severity": 1,
                # missing cs_style + cs_authenticity
            }
        ],
    )
    with pytest.raises(DatasetError):
        load_dataset(bad)


def test_en_row_with_style_rejected(tmp_path):
    bad = tmp_path / "bad.jsonl"
    write_jsonl(
        bad,
        [
            {
                "id": "X-001-EN",
                "base_id": "X-001",
                "harm_category": "H1",
                "condition": "EN",
                "cs_style": "A",  # not allowed for EN
                "prompt": "hello",
                "harm_severity": 1,
            }
        ],
    )
    with pytest.raises(DatasetError):
        load_dataset(bad)


def test_duplicate_id_rejected(tmp_path):
    bad = tmp_path / "dup.jsonl"
    row = {
        "id": "H1-001-EN",
        "base_id": "H1-001",
        "harm_category": "H1",
        "condition": "EN",
        "prompt": "hello",
        "harm_severity": 1,
    }
    write_jsonl(bad, [row, row])
    with pytest.raises(DatasetError):
        load_dataset(bad)


def test_invalid_harm_category_rejected(tmp_path):
    bad = tmp_path / "bad.jsonl"
    write_jsonl(
        bad,
        [
            {
                "id": "Z-001-EN",
                "base_id": "Z-001",
                "harm_category": "H9",  # invalid
                "condition": "EN",
                "prompt": "hello",
                "harm_severity": 1,
            }
        ],
    )
    with pytest.raises(DatasetError):
        load_dataset(bad)


def test_invalid_severity_rejected(tmp_path):
    bad = tmp_path / "bad.jsonl"
    write_jsonl(
        bad,
        [
            {
                "id": "Y-001-EN",
                "base_id": "Y-001",
                "harm_category": "H1",
                "condition": "EN",
                "prompt": "hello",
                "harm_severity": 5,  # >3
            }
        ],
    )
    with pytest.raises(DatasetError):
        load_dataset(bad)


def test_whitespace_prompt_rejected(tmp_path):
    bad = tmp_path / "ws.jsonl"
    write_jsonl(
        bad,
        [
            {
                "id": "W-001-EN",
                "base_id": "W-001",
                "harm_category": "H1",
                "condition": "EN",
                "prompt": "   ",
                "harm_severity": 1,
            }
        ],
    )
    with pytest.raises(DatasetError):
        load_dataset(bad)


def test_pairing_coverage_counts():
    rows = load_dataset(FIXTURE)
    cov = pairing_coverage(rows)
    # H1-001 has EN + UR + CS + SM in the fixture
    assert cov["H1-001"] == {"EN": 1, "UR": 1, "CS": 1, "SM": 1}
    # H2-001 has EN + UR + 2 CS, no SM
    assert cov["H2-001"]["CS"] == 2
    assert cov["H2-001"]["SM"] == 0
