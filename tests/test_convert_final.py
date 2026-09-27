"""Acceptance checks 1-3: final-schema ingestion, identity, preservation."""
from __future__ import annotations

import pytest

from csjail.convert_final import convert, id_namespace
from csjail.data import Prompt, load_dataset, split_row_id, validate_structure
from csjail.utils.io import write_jsonl
from tests.conftest import FIXTURE_CSV, REAL_CSV, read_fixture_csv, write_csv


def test_fixture_converts_to_four_rows_per_family(fixture_rows):
    assert len(fixture_rows) == 48
    rep = validate_structure(fixture_rows)
    assert rep.n_families == 12 and rep.n_complete == 12


def test_ids_are_opaque_strings_and_preserved(fixture_rows):
    fams = {r.base_id for r in fixture_rows}
    assert "0012" in fams and "12" not in fams          # never int-cast
    assert "CSJUR-R-0246-01" in fams
    r = next(r for r in fixture_rows if r.base_id == "CSJUR-V3-0001" and r.condition == "RU")
    assert r.id == "CSJUR-V3-0001::RU"
    assert split_row_id(r.id) == ("CSJUR-V3-0001", "RU")


def test_domain_not_in_identity(fixture_rows):
    assert all(r.domain_id not in r.base_id for r in fixture_rows)


def test_newlines_preserved_and_normalization_recorded(fixture_rows):
    en = next(r for r in fixture_rows if r.base_id == "0012" and r.condition == "EN")
    assert "\nSecond line kept." in en.prompt
    cs = next(r for r in fixture_rows if r.base_id == "31" and r.condition == "CS")
    assert cs.prompt == cs.prompt.strip()
    assert cs.provenance.text_normalization == "strip_outer_whitespace"
    assert cs.provenance.source_text.startswith("  ")


def test_no_fabricated_metadata(fixture_rows):
    for r in fixture_rows:
        assert r.harm_severity is None and r.cs_style is None and r.cs_authenticity is None
        assert r.provenance.author is None and r.provenance.model_assistance is None


def test_id_namespace_is_descriptive_only():
    assert id_namespace("42") == ("numeric", None)
    assert id_namespace("CSJUR-V4-0003") == ("CSJUR-V4", None)
    assert id_namespace("CSJUR-X1-0062") == ("CSJUR-X1", None)
    ns, hint = id_namespace("CSJUR-R-0246-01")
    assert ns == "CSJUR-R" and hint == "id-references-replaced-prompt:0246"


def test_metadata_round_trip(tmp_path, fixture_rows):
    rows = [r.model_copy(update={"qa": {"status": "open", "reviewer": "R1"}}) for r in fixture_rows]
    p = tmp_path / "d.jsonl"
    write_jsonl(p, [r.model_dump() for r in rows])
    back = load_dataset(p)
    assert back[0].qa == {"status": "open", "reviewer": "R1"}
    assert back[0].domain_name == rows[0].domain_name
    assert back[0].provenance == rows[0].provenance


@pytest.mark.parametrize("mutate,msg", [
    (lambda rs: rs[0].update(domain_name="Wrong name"), "domain_name"),
    (lambda rs: rs[0].update(domain_id="D9"), "unknown domain_id"),
    (lambda rs: rs[1].update(CS="   "), "blank CS"),
    (lambda rs: rs[2].update(prompt_id=rs[3]["prompt_id"]), "duplicate prompt_id"),
    (lambda rs: rs[0].update(prompt_id="a::b"), "reserved"),
])
def test_bad_input_fails(tmp_path, mutate, msg):
    rows = read_fixture_csv()
    mutate(rows)
    with pytest.raises(ValueError, match=msg):
        convert(write_csv(tmp_path / "bad.csv", rows))


def test_wrong_columns_fail(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("prompt_id,category,EN,CS,RU,UR\n1,1,a,b,c,d\n", encoding="utf-8")
    with pytest.raises(ValueError, match="required columns"):
        convert(p)


def test_new_family_changes_counts_dynamically(tmp_path):
    rows = read_fixture_csv()
    rows.append({**rows[0], "prompt_id": "CSJUR-V9-0001", "EN": "new en", "CS": "new cs",
                 "RU": "new ru", "UR": "نیا"})
    out, rep = convert(write_csv(tmp_path / "plus.csv", rows))
    assert rep["n_families"] == 13 and len(out) == 52


def test_text_edit_changes_content_hash(tmp_path):
    rows = read_fixture_csv()
    _, a = convert(write_csv(tmp_path / "a.csv", rows))
    rows[5]["RU"] = rows[5]["RU"] + " edited"
    _, b = convert(write_csv(tmp_path / "b.csv", rows))
    assert a["normalized_content_sha256"] != b["normalized_content_sha256"]
    assert a["dataset_version"] != b["dataset_version"]


@pytest.mark.skipif(not REAL_CSV.exists(), reason="final dataset is local-only (gitignored)")
def test_real_snapshot_acceptance():
    rows, rep = convert(REAL_CSV)
    prompts = [Prompt.model_validate(r) for r in rows]
    validate_structure(prompts)
    assert rep["n_families"] == 692 and rep["n_long_rows"] == 2768
    assert rep["families_by_domain"] == {"D1": 122, "D2": 108, "D3": 112, "D4": 118,
                                         "D5": 112, "D6": 120}
    assert rep["families_by_id_namespace"] == {"CSJUR-R": 18, "CSJUR-V3": 100,
                                               "CSJUR-V4": 84, "numeric": 490}
    assert rep["review_flags"]["equal_variants_within_family"] == []
    assert rep["review_flags"]["exact_duplicates_across_families"] == []
    ur_latin = [f for f in rep["review_flags"]["script"] if f["flag"] == "ur_contains_latin"]
    assert len(ur_latin) == 60


def _ten_column(rows, **overrides):
    out = []
    for i, r in enumerate(rows, 1):
        out.append({"row_number": str(i), **r, "evaluation_stratum": "harmful",
                    "approval_status": "approved", **overrides.get(r["prompt_id"], {})})
    return out


def _write_any(path, rows):
    import csv
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    return path


def test_ten_column_release_preserves_metadata(tmp_path):
    rows = _ten_column(read_fixture_csv())
    out, rep = convert(_write_any(tmp_path / "ten.csv", rows))
    assert rep["extra_columns_preserved"] == ["row_number", "evaluation_stratum", "approval_status"]
    assert rep["metadata_value_counts"]["approval_status"] == {"approved": 12}
    r = next(x for x in out if x["base_id"] == "0012" and x["condition"] == "CS")
    assert r["provenance"]["source_metadata"] == {"row_number": "2", "evaluation_stratum": "harmful",
                                                   "approval_status": "approved"}
    assert r["id"] == "0012::CS"                       # row_number is never identity
    seven, _ = convert(FIXTURE_CSV)                     # same content -> same dataset version
    assert out[0]["dataset_version"] == seven[0]["dataset_version"]


@pytest.mark.parametrize("col,val", [("approval_status", "pending"),
                                     ("evaluation_stratum", "benign")])
def test_unsupported_metadata_values_fail(tmp_path, col, val):
    rows = _ten_column(read_fixture_csv(), **{"31": {col: val}})
    with pytest.raises(ValueError, match="unsupported"):
        convert(_write_any(tmp_path / "bad.csv", rows))
