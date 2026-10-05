"""Stage C driver: input discovery, step gating, time budget, always-packaged resume bundle."""
from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load():
    spec = importlib.util.spec_from_file_location("stagec_run", ROOT / "scripts" / "stagec_run.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["stagec_run"] = mod
    spec.loader.exec_module(mod)
    return mod


ADJ = "outputs/exp1/validation-kaggle-01-merged/adjudication.csv"
JUDGMENTS = "outputs/exp2/main/judgments.jsonl"


def bundle(root: Path, marker: str, stamp: str, files: dict | None = None) -> Path:
    (root / "outputs" / "exp1" / "validation-kaggle-01-merged").mkdir(parents=True)
    for rel, text in (files or {}).items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text, encoding="utf-8")
    (root / marker).write_text(stamp, encoding="utf-8")
    return root


def test_inputs_newest_judgments_and_newest_human_labels_win(tmp_path):
    sc = load()
    inp = tmp_path / "input"
    bundle(inp / "upload", sc.UPLOAD, "", {ADJ: "corrected"})             # re-uploaded, newer labels
    bundle(inp / "old", sc.RESUME, "2026-10-05T00:00:00Z", {JUDGMENTS: "old", ADJ: "original"})
    newest = bundle(tmp_path / "build", sc.RESUME, "2026-10-06T00:00:00Z",
                    {JUDGMENTS: "new", ADJ: "original"})
    (inp / "prev").mkdir()
    with zipfile.ZipFile(inp / "prev" / "stageC_outputs.zip", "w") as z:   # previous output, zipped
        for f in newest.rglob("*"):
            if f.is_file():
                z.write(f, f.relative_to(newest).as_posix())
    dest = tmp_path / "repo"
    used = sc.collect_inputs(inp, tmp_path / "scratch", dest)
    assert [Path(u).name for u in used] == ["old", "stageC_outputs", "upload"]
    assert (dest / JUDGMENTS).read_text(encoding="utf-8") == "new"
    assert (dest / ADJ).read_text(encoding="utf-8") == "corrected"


def run_driver(sc, tmp_path, monkeypatch, codes, *extra):
    repo, inp = tmp_path / "repo", tmp_path / "input"
    repo.mkdir()
    bundle(inp / "upload", sc.UPLOAD, "", {"outputs/exp2/main/generations.jsonl": "gens"})
    monkeypatch.setattr(sc, "ROOT", repo)
    calls, passed = [], {"v": False}

    def fake_step(cmd, timeout_s):
        name = next(n for n, c, _ in sc.plan(True) if c == cmd)
        calls.append(name)
        if name == "harm_gate" and codes.get(name) == 0:
            passed["v"] = True
        return codes.get(name, 0)

    monkeypatch.setattr(sc, "run_step", fake_step)
    monkeypatch.setattr(sc, "harm_passed", lambda: passed["v"])
    out = tmp_path / "out" / "stageC_outputs.zip"
    out.parent.mkdir()
    rc = sc.main(["--input-root", str(inp), "--scratch", str(tmp_path / "scratch"),
                  "--out-zip", str(out), *extra])
    with zipfile.ZipFile(out) as z:
        status = json.loads(z.read("STAGEC_STATUS.json"))
        names = set(z.namelist())
    return rc, calls, status, names


def test_exp2_judged_only_after_pass_and_robustness_only_after_main(tmp_path, monkeypatch):
    sc = load()
    rc, calls, status, names = run_driver(sc, tmp_path, monkeypatch,
                                          {"harm_gate": 0, "exp2_main": 4})
    assert rc == 0 and calls == ["harm_gate", "exp2_main"]          # benign absent, robustness waits
    assert status["steps"]["exp2_main"] == 4
    assert status["steps"]["exp2_robustness"].startswith("skipped")
    assert sc.RESUME in names and "outputs/exp2/main/generations.jsonl" in names


def test_failed_gate_blocks_exp2_judging(tmp_path, monkeypatch):
    sc = load()
    rc, calls, status, _ = run_driver(sc, tmp_path, monkeypatch, {"harm_gate": 1})
    assert calls == ["harm_gate"] and status["steps"]["exp2_main"].startswith("skipped")


def test_time_budget_skips_work_but_still_packages(tmp_path, monkeypatch):
    sc = load()
    rc, calls, status, names = run_driver(sc, tmp_path, monkeypatch, {"harm_gate": 0},
                                          "--time-budget-h", "0.01")
    assert calls == [] and status["steps"]["harm_gate"] == "skipped: time budget used up"
    assert sc.RESUME in names


def test_missing_inputs_still_package_and_report(tmp_path, monkeypatch):
    sc = load()
    monkeypatch.setattr(sc, "ROOT", tmp_path / "repo")
    (tmp_path / "repo").mkdir()
    out = tmp_path / "o.zip"
    assert sc.main(["--input-root", str(tmp_path / "nothing"), "--scratch", str(tmp_path / "s"),
                    "--out-zip", str(out)]) == 1
    with zipfile.ZipFile(out) as z:
        assert "no input bundle" in json.loads(z.read("STAGEC_STATUS.json"))["errors"][0]


def test_bundle_round_trips_as_next_input(tmp_path, monkeypatch):
    sc = load()
    _, _, _, _ = run_driver(sc, tmp_path, monkeypatch, {"harm_gate": 0, "exp2_main": 0})
    nxt = tmp_path / "next_input"
    nxt.mkdir()
    (tmp_path / "out" / "stageC_outputs.zip").rename(nxt / "stageC_outputs.zip")
    dest = tmp_path / "repo2"
    used = sc.collect_inputs(nxt, tmp_path / "scratch2", dest)
    assert len(used) == 1 and (dest / "outputs/exp2/main/generations.jsonl").exists()
