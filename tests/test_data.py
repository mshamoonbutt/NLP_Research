"""Schema + exact structural contract (acceptance checks 2-3)."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from csjail.data import (
    DatasetError, Prompt, check_structure, load_dataset, summarize, validate_structure,
)
from csjail.utils.io import write_jsonl
from tests.conftest import ROOT, family


def test_committed_fixture_loads_and_is_complete():
    rows = load_dataset(ROOT / "data" / "csjail_fixture.jsonl")
    assert validate_structure(rows).n_complete == 12


def test_exactly_one_row_per_condition():
    rows = family("1") + family("2")
    assert check_structure(rows).ok
    cs = next(r for r in rows if r.condition == "CS")
    dup = rows + [cs.model_copy(update={"id": "1::CS-dup"})]
    rep = check_structure(dup)
    assert not rep.ok and any("expected exactly 1 CS" in e for e in rep.errors)


def test_missing_condition_fails():
    rows = [r for r in family("1") if r.condition != "RU"]
    with pytest.raises(DatasetError, match="exactly 1 RU"):
        validate_structure(rows)


def test_inconsistent_domain_fails():
    rows = family("1", "D1")
    rows[2] = rows[2].model_copy(update={"domain_id": "D2", "harm_category": "D2"})
    rep = check_structure(rows)
    assert any("inconsistent domain_id" in e for e in rep.errors)


def test_legacy_sm_rejected_by_default(tmp_path):
    legacy = ROOT / "tests" / "fixtures" / "legacy" / "csjail_fixture_v0.jsonl"
    with pytest.raises(DatasetError, match="legacy"):
        load_dataset(legacy)
    assert load_dataset(legacy, allow_legacy=True)


def test_unknown_fields_rejected_not_dropped():
    with pytest.raises(ValidationError):
        Prompt(id="1::EN", base_id="1", condition="EN", prompt="x", harm_category="D1",
               domain_id="D1", surprise="field")


def test_no_annotation_fields_required():
    p = Prompt(id="1::EN", base_id="1", condition="EN", prompt="x", domain_id="D1")
    assert p.harm_category == "D1" and p.harm_severity is None


def test_alias_must_match_domain():
    with pytest.raises(ValidationError):
        Prompt(id="1::EN", base_id="1", condition="EN", prompt="x", domain_id="D1",
               harm_category="D2")


def test_summary_counts_from_data(tmp_path):
    rows = family("1", "D1") + family("2", "D2") + family("3", "D2")
    s = summarize(rows)
    assert s["n_families"] == 3 and s["n_rows"] == 12
    assert s["families_by_domain"] == {"D1": 1, "D2": 2}
    p = tmp_path / "x.jsonl"
    write_jsonl(p, [r.model_dump() for r in rows])
    assert len(load_dataset(p)) == 12
