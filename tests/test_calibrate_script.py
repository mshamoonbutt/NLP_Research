"""calibrate_judge.py: development samples give a report, never a manifest;
validation samples write the manifest. Uses saved predictions (no API)."""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

from csjail.judge import load_judge_config

ROOT = Path(__file__).resolve().parent.parent


def calibrate():
    spec = importlib.util.spec_from_file_location("calib", ROOT / "scripts" / "calibrate_judge.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["calib"] = mod
    spec.loader.exec_module(mod)
    return mod.main


def write(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def make_sample(d: Path, role: str) -> Path:
    d.mkdir()
    fp = load_judge_config().fingerprint("harm")["fingerprint_id"]
    items, r1, r2, preds = [], [], [], []
    for c in ("CS", "EN", "RU", "UR"):
        for i in range(24):
            iid = f"{role}:m{i % 2}:{c}{i}"
            u = i % 2 == 0
            k = "harmful_compliance" if u else "refusal"
            items.append({"item_id": iid, "model": f"m{i % 2}", "condition": c,
                          "sample_kind": "representative", "prompt": "p", "response": "r"})
            r1.append({"item_id": iid, "rater_id": "A", "unsafe": u, "response_kind": k})
            r2.append({"item_id": iid, "rater_id": "B", "unsafe": u, "response_kind": k})
            preds.append({"item_id": iid, "judge_fingerprint_id": fp, "judge_status": "ok",
                          "judge_unsafe": u, "judge_response_kind": k})
    write(d / "items.csv", list(items[0]), items)
    write(d / "rater1.csv", list(r1[0]), r1)
    write(d / "rater2.csv", list(r2[0]), r2)
    write(d / "adjudication.csv", ["item_id", "gold_unsafe", "gold_response_kind", "resolution"], [])
    (d / "sample_manifest.json").write_text(json.dumps({
        "role": role, "sample_kind": "representative", "dataset_version": "v", "split_id": "s",
        "seed": 1, "models": ["m0", "m1"], "n_items": len(items), "families": ["f"],
        "items_sha256": "x"}), encoding="utf-8")
    with (d / "preds.jsonl").open("w", encoding="utf-8") as f:
        for p in preds:
            f.write(json.dumps(p) + "\n")
    return d


def test_development_sample_reports_without_manifest(tmp_path):
    d = make_sample(tmp_path / "dev", "development")
    man = tmp_path / "manifest.json"
    rc = calibrate()(["--sample-dir", str(d), "--predictions-from", str(d / "preds.jsonl"),
                      "--manifest-out", str(man)])
    assert rc == 0 and not man.exists()
    [rep] = list(d.glob("development_report_harm_*.json"))
    body = json.loads(rep.read_text(encoding="utf-8"))
    assert body["not_a_gate"] is True and body["result"]["status"] == "PASS"


def test_validation_sample_writes_manifest(tmp_path):
    d = make_sample(tmp_path / "val", "validation")
    man = tmp_path / "manifest.json"
    rc = calibrate()(["--sample-dir", str(d), "--predictions-from", str(d / "preds.jsonl"),
                      "--manifest-out", str(man)])
    assert rc == 0 and json.loads(man.read_text(encoding="utf-8"))["status"] == "PASS"
