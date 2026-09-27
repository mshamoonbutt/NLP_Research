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
    assert man["gates"] == {"ingest": "PASS", "structure": "PASS", "split": "PASS",
                            "exposure": "PASS"}
    assert (d / "FINDINGS.md").exists() and (d / "exposure_report.json").exists()
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


def test_restore_regenerates_only_matching_dataset(tmp_path):
    out = tmp_path / "exp0"
    base = ["--source-csv", str(FIXTURE_CSV), "--eval-size", "4", "--out-root", str(out),
            "--no-latest"]
    assert exp0()(base) == 0
    [d] = [p for p in out.iterdir() if p.name.startswith("final-")]
    raw = (d / "dataset_final.jsonl").read_bytes()
    assert b"\r\n" not in raw                      # LF on every OS -> stable hash
    (d / "dataset_final.jsonl").unlink()           # as on a fresh clone (gitignored)
    assert exp0()(base + ["--restore"]) == 0
    assert (d / "dataset_final.jsonl").read_bytes() == raw
    resolve_exp0(d).load_rows()
    assert exp0()(["--source-csv", str(FIXTURE_CSV), "--eval-size", "5", "--out-root", str(out),
                   "--no-latest", "--restore"]) in (0, 5)   # different split -> new version dir or fail


def exp0_module():
    spec = importlib.util.spec_from_file_location("exp0m", ROOT / "scripts" / "exp0_finalize_data.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _core(tmp_path):
    out = tmp_path / "exp0"
    assert exp0()(["--source-csv", str(FIXTURE_CSV), "--eval-size", "4", "--out-root", str(out),
                   "--no-latest", "--duplicate-decisions", str(tmp_path / "none.csv")]) == 0
    d = next(p for p in out.iterdir() if p.name.startswith("final-"))
    return out, d, json.loads((d / "split_manifest.json").read_text(encoding="utf-8"))


def _sample(tmp_path, name, families, role="development"):
    s = tmp_path / name
    s.mkdir()
    (s / "sample_manifest.json").write_text(json.dumps({"role": role, "families": families}),
                                            encoding="utf-8")
    return str(s)


def test_exposure_gate(tmp_path):
    out, d, split = _core(tmp_path)
    ev = sorted(f for f, a in split["assignments"].items() if a["split"] == "eval_main")
    tr = sorted(f for f, a in split["assignments"].items() if a["split"] == "train_pool")
    ok = _sample(tmp_path, "dev", tr[:2])
    pilot = _sample(tmp_path, "pilot", tr[1:4], role="feasibility")
    out2 = tmp_path / "exp0b"
    assert exp0()(["--source-csv", str(FIXTURE_CSV), "--extend-split", str(d / "split_manifest.json"),
                   "--out-root", str(out2), "--no-latest", "--exposure-samples", ok, pilot]) == 0
    rep = json.loads((next(out2.iterdir()) / "exposure_report.json").read_text(encoding="utf-8"))
    assert rep["status"] == "PASS" and rep["n_exposed_families"] == 4
    assert rep["pairwise_family_overlap"] == {f"{ok} & {pilot}": 1}
    bad = _sample(tmp_path, "bad", [ev[0]])
    out3 = tmp_path / "exp0c"
    assert exp0()(["--source-csv", str(FIXTURE_CSV), "--extend-split", str(d / "split_manifest.json"),
                   "--out-root", str(out3), "--no-latest", "--exposure-samples", bad]) == 6
    assert next(out3.iterdir()).name.startswith("FAILED-")


def _near_copy(rows, fam, new_id):
    src = next(r for r in rows if r["prompt_id"] == fam)
    return {**src, "prompt_id": new_id, "EN": src["EN"] + " kindly", "CS": src["CS"] + " please",
            "RU": src["RU"] + " zara", "UR": src["UR"] + " جلدی"}


def test_new_family_screen_quarantines_eval_relatives(tmp_path):
    out, d, split = _core(tmp_path)
    ev = sorted(f for f, a in split["assignments"].items() if a["split"] == "eval_main")[0]
    rows = read_fixture_csv()
    rows.append(_near_copy(rows, ev, "CSJUR-X9-0001"))
    csv_path = write_csv(tmp_path / "plus.csv", rows)
    out2 = tmp_path / "e2"
    assert exp0()(["--source-csv", str(csv_path), "--extend-split", str(d / "split_manifest.json"),
                   "--out-root", str(out2), "--no-latest"]) == 0
    first = next(out2.iterdir())
    a = json.loads((first / "split_manifest.json").read_text(encoding="utf-8"))
    assert a["assignments"]["CSJUR-X9-0001"]["split"] == "excluded_group_relative"
    flagged = json.loads((first / "exp0_report.json").read_text(encoding="utf-8"))[
        "new_family_screen"]["flagged"]
    assert any(p["family_b"] == ev for p in flagged)   # templated fixture: several matches
    dec = tmp_path / "dec.csv"
    dec.write_text("family_a,family_b,decision\n" + "".join(
        f"{p['family_a']},{p['family_b']},distinct\n" for p in flagged), encoding="utf-8")
    out3 = tmp_path / "e3"
    assert exp0()(["--source-csv", str(csv_path), "--extend-split", str(d / "split_manifest.json"),
                   "--out-root", str(out3), "--no-latest", "--duplicate-decisions", str(dec)]) == 0
    a = json.loads((next(out3.iterdir()) / "split_manifest.json").read_text(encoding="utf-8"))
    assert a["assignments"]["CSJUR-X9-0001"]["split"] == "train_pool"


def test_training_only_extension(tmp_path):
    out, d, split = _core(tmp_path)
    ev = sorted(f for f, a in split["assignments"].items() if a["split"] == "eval_main")[0]
    base = read_fixture_csv()
    ext = [_near_copy(base, ev, "CSJUR-T1-0002"),
           {**base[0], "prompt_id": "CSJUR-T1-0001", "EN": "a brand new harmless request",
            "CS": "bilkul naya harmless sawal", "RU": "bilkul naya sawal", "UR": "بالکل نیا سوال"}]
    ext_csv = write_csv(tmp_path / "ext.csv", ext)
    rc = exp0()(["--training-extension-csv", str(ext_csv), "--extension-name", "t1",
                 "--core-exp0-dir", str(d), "--out-root", str(out), "--exposure-samples"])
    assert rc == 0 and not (out / "LATEST.json").exists()
    e = next(p for p in out.iterdir() if "+ext-t1-" in p.name)
    man = json.loads((e / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert man["training_extension"]["n_added"] == 2 and man["training_extension"]["n_quarantined"] == 1
    a = json.loads((e / "split_manifest.json").read_text(encoding="utf-8"))["assignments"]
    assert a["CSJUR-T1-0001"] == {**a["CSJUR-T1-0001"], "split": "train_pool", "extension": "t1"}
    assert a["CSJUR-T1-0002"]["split"] == "excluded_group_relative"
    assert all(a[f]["split"] == x["split"] for f, x in split["assignments"].items())
    assert len(resolve_exp0(e).load_rows()) == 56
    clash = write_csv(tmp_path / "clash.csv", [base[0]])
    assert exp0()(["--training-extension-csv", str(clash), "--extension-name", "t2",
                   "--core-exp0-dir", str(d), "--out-root", str(out), "--exposure-samples"]) == 1


def test_release_registry_checksum(tmp_path, monkeypatch):
    mod = exp0_module()
    reg = tmp_path / "dataset.yaml"
    reg.write_text(json.dumps({"active": "fx", "releases": {"fx": {
        "source_csv": str(FIXTURE_CSV), "sha256": "0" * 64}}, "exposure_samples": []}),
        encoding="utf-8")
    monkeypatch.setattr(mod, "REGISTRY", reg)
    assert mod.main(["--eval-size", "4", "--out-root", str(tmp_path / "o"), "--no-latest"]) == 1
    from csjail.utils.io import sha256_file
    reg.write_text(json.dumps({"active": "fx", "releases": {"fx": {
        "source_csv": str(FIXTURE_CSV), "sha256": sha256_file(FIXTURE_CSV)}},
        "exposure_samples": []}), encoding="utf-8")
    assert mod.main(["--eval-size", "4", "--out-root", str(tmp_path / "o2"), "--no-latest"]) == 0
    man = json.loads((next((tmp_path / "o2").iterdir()) / "dataset_manifest.json").read_text(
        encoding="utf-8"))
    assert man["release"] == "fx"
