"""exp1_dev_gold: merge by item_id, human decisions override, bad decisions rejected."""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def mod():
    spec = importlib.util.spec_from_file_location("devgold", ROOT / "scripts" / "exp1_dev_gold.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["devgold"] = m
    spec.loader.exec_module(m)
    return m


def w(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        d = csv.DictWriter(f, fieldnames=list(rows[0]))
        d.writeheader()
        d.writerows(rows)


def test_dev_gold(tmp_path):
    a = tmp_path / "annotations"
    a.mkdir()
    ids = ["i1", "i2", "i3"]
    w(tmp_path / "items.csv", [{"item_id": i, "model": "m", "condition": "CS", "base_id": i,
                                "domain_id": "D1"} for i in ids])
    lab = {"unsafe": "true", "response_kind": "harmful_compliance", "compliance_degree": "partial"}
    r1 = [{"item_id": i, "rater_id": "AI", **lab, "notes": ""} for i in reversed(ids)]  # other order
    r2 = [{"item_id": i, "rater_id": "AI2", **lab, "notes": ""} for i in ids]
    r2[2] = {**r2[2], "unsafe": "false", "response_kind": "refusal", "compliance_degree": "none"}
    w(a / "rater1_x.csv", r1)
    w(a / "rater2_x.csv", r2)
    blank = {"gold_unsafe": "", "gold_response_kind": "", "gold_compliance_degree": "",
             "decision_rule": "", "adjudicator": ""}
    w(a / "adjudication_questions.csv", [
        {"item_id": "i1", **blank, "gold_unsafe": "false", "gold_response_kind": "refusal",
         "gold_compliance_degree": "none", "adjudicator": "V1"},
        {"item_id": "i2", **blank}])
    d = mod().build(tmp_path)
    gold = {r["item_id"]: r for r in csv.DictReader(open(tmp_path / "development_gold.csv",
                                                        encoding="utf-8-sig"))}
    assert gold["i1"]["gold_status"] == "human_adjudicated" and gold["i1"]["gold_unsafe"] == "False"
    assert gold["i2"]["gold_status"] == "pending_adjudication" and gold["i2"]["gold_unsafe"] == "True"
    assert gold["i3"]["gold_status"] == "pending_disagreement"
    assert d["independent_annotations"] is None and d["n_items_labels_identical"] == 2
    meta = json.loads((a / "verification_metadata.json").read_text(encoding="utf-8"))
    assert meta["human_verification"]["human_verifier_ids"] is None
    meta["human_verification"]["human_verifier_ids"] = ["V1"]
    (a / "verification_metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    mod().build(tmp_path)   # human fields survive a re-run
    assert json.loads((a / "verification_metadata.json").read_text(encoding="utf-8"))[
        "human_verification"]["human_verifier_ids"] == ["V1"]
    w(a / "adjudication_questions.csv", [{"item_id": "i2", **blank, "gold_unsafe": "true",
                                          "gold_response_kind": "refusal",
                                          "gold_compliance_degree": "none", "adjudicator": "V1"}])
    with pytest.raises(ValueError, match="inconsistent"):
        mod().build(tmp_path)
