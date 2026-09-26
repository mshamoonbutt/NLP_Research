"""Exp 0 finalization: temp-then-publish, frozen outputs, hash checks (checks 3, 5)."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from csjail.artifacts import ArtifactError, resolve_exp0
from tests.conftest import FIXTURE_CSV, read_fixture_csv, write_csv

ROOT = Path(__file__).resolve().parent.parent


def exp0():
    spec = importlib.util.spec_from_file_location("exp0", ROOT / "scripts" / "exp0_finalize_data.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.main


def test_finalize_fixture(tmp_path):
    out = tmp_path / "exp0"
    rc = exp0()(["--source-csv", str(FIXTURE_CSV), "--eval-size", "4", "--out-root", str(out),
                 "--no-latest", "--duplicate-decisions", str(tmp_path / "none.csv")])
    assert rc == 0
    [d] = [p for p in out.iterdir() if p.name.startswith("final-")]
    assert (d / "FINALIZED").exists()
    man = json.loads((d / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert man["gates"] == {"ingest": "PASS", "structure": "PASS", "split": "PASS"}
    assert "kappa" not in json.dumps(man["gates"])
    art = resolve_exp0(d)
    rows = art.load_rows()
    assert len(rows) == 48 and all(r.group_id for r in rows)
    assert all(r.cmi is None and r.cmi_heuristic is not None for r in rows)
    flags = json.loads((d / "qa_review_flags.json").read_text(encoding="utf-8"))
    assert "prompt" not in json.dumps(flags)          # IDs only, no text
    # frozen: a second run refuses to overwrite
    assert exp0()(["--source-csv", str(FIXTURE_CSV), "--eval-size", "4", "--out-root", str(out),
                   "--no-latest"]) == 4
    # tampering is detected
    ds = d / "dataset_final.jsonl"
    ds.write_text(ds.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ArtifactError, match="changed after finalization"):
        resolve_exp0(d)


def test_failure_leaves_diagnostics_only(tmp_path):
    rows = read_fixture_csv()
    rows[0]["UR"] = ""
    bad = write_csv(tmp_path / "bad.csv", rows)
    out = tmp_path / "exp0"
    assert exp0()(["--source-csv", str(bad), "--out-root", str(out), "--no-latest"]) == 1
    [d] = list(out.iterdir())
    assert d.name.startswith("FAILED-") and (d / "DIAGNOSTIC_ONLY").exists()
    assert not (d / "FINALIZED").exists() and not (out / "LATEST.json").exists()
    with pytest.raises(ArtifactError):
        resolve_exp0(d)


def test_extend_split_append_only(tmp_path):
    out = tmp_path / "exp0"
    assert exp0()(["--source-csv", str(FIXTURE_CSV), "--eval-size", "4", "--out-root", str(out),
                   "--no-latest"]) == 0
    [d] = [p for p in out.iterdir() if p.name.startswith("final-")]
    old = json.loads((d / "split_manifest.json").read_text(encoding="utf-8"))
    rows = read_fixture_csv()
    rows.append({**rows[0], "prompt_id": "CSJUR-V9-0001", "EN": "new en", "CS": "new cs",
                 "RU": "new ru", "UR": "نیا"})
    new_csv = write_csv(tmp_path / "plus.csv", rows)
    assert exp0()(["--source-csv", str(new_csv), "--out-root", str(out), "--no-latest",
                   "--extend-split", str(d / "split_manifest.json")]) == 0
    [d2] = [p for p in out.iterdir() if p.name.startswith("final-13-")]
    new = json.loads((d2 / "split_manifest.json").read_text(encoding="utf-8"))
    for f, a in old["assignments"].items():
        assert new["assignments"][f]["split"] == a["split"]
    assert new["assignments"]["CSJUR-V9-0001"]["added_after_freeze"] is True
