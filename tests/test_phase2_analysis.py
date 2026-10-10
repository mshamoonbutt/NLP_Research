"""Phase 2 analysis (scripts/phase2_analysis.py) and post-training audit (scripts/exp8_audit.py)
on synthetic judged Exp 8 folders: Llama has a real C-vs-B_ext effect on CS, Phi-3 has none."""
from __future__ import annotations

import collections
import csv
import importlib.util
import json
import shutil
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
FP = "fp-test"
FAMS = [str(i) for i in range(1, 61)]                      # D1 = 1..30, D2 = 31..60
BUDGET = {"phi3": (39, 30), "llama32": (60, 51)}


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def dom(f):
    return "D1" if int(f) <= 30 else "D2"


def unsafe(model, arm, cond, f, tag):
    i = int(f)
    if cond != "CS":
        return i % 7 == 0 and arm in ("A", "E")
    if model == "phi3":
        return i % 5 == 0 if arm != "A" else i % 20 == 0          # C == B_ext: no effect; A: 3 flagged
    return {"A": i % 2 == 0, "E": i % 3 == 0, "B_ext": i % 2 == 0, "C": i % 10 == 0}[arm]


def rec(model, arm, cond, f, tag, probe=False):
    r = {"model": model, "arm": arm, "condition": cond, "base_id": f"BN-{f}" if probe else f,
         "row_id": f"probe::BN-{f}::{cond}" if probe else f"{f}::{cond}", "domain_id": dom(f),
         "judge_status": "ok", "generation_status": "ok", "response": f"response {len(f) * 7}",
         "response_sha256": f"{tag}|{arm}|{f}|{cond}|{probe}"}
    if probe:
        r.update(probe=True, judge_rubric_kind="benign", judge_refused=int(f) % 6 == 0 or (arm == "C" and int(f) % 4 == 0),
                 judge_fingerprint_id="fp-benign")
    else:
        r.update(judge_unsafe=unsafe(model, arm, cond, f, tag), judge_fingerprint_id=FP)
    return r


@pytest.fixture
def fake_root(tmp_path):
    (tmp_path / "configs").mkdir()
    cfg = yaml.safe_load((ROOT / "configs" / "dpo.yaml").read_text(encoding="utf-8"))
    cfg["analysis"]["permutation_n"] = 4000
    (tmp_path / "configs" / "dpo.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    ev = yaml.safe_load((ROOT / "configs" / "eval.yaml").read_text(encoding="utf-8"))
    ev["stats"]["bootstrap_n"] = 2000
    (tmp_path / "configs" / "eval.yaml").write_text(yaml.safe_dump(ev), encoding="utf-8")
    (tmp_path / "outputs/exp6/review").mkdir(parents=True)
    (tmp_path / "outputs/exp6/review/verified_summary.json").write_text(json.dumps(
        {"budget_main": {m: b[0] for m, b in BUDGET.items()}, "budget_ablation": {m: b[1] for m, b in BUDGET.items()}}))
    (tmp_path / "outputs/exp9/ablation_D2").mkdir(parents=True)
    (tmp_path / "outputs/exp9/ablation_D2/split_manifest.json").write_text(json.dumps({"meta": {"ablation_domain": "D2"}}))
    counts = {"tp": 30, "fn": 5, "tn": 95, "fp": 10}
    cells = {f"{m}|{c}": dict(counts) for m in BUDGET for c in ("EN", "CS", "RU", "UR")}
    cells["phi3|CS"] = {"tp": 14, "fn": 1, "tn": 125, "fp": 0}
    (tmp_path / "outputs/exp1").mkdir(parents=True)
    (tmp_path / "outputs/exp1/judge_validation_manifest.json").write_text(json.dumps({
        "judge_fingerprint": {"fingerprint_id": FP},
        "result": {"per_language": {c: dict(counts) for c in ("EN", "CS", "RU", "UR")}, "per_model_language": cells}}))
    for model, (n, nab) in BUDGET.items():
        tags = [f"n{n}", f"n{n}_s43", f"n{n}_s44"] + [f"n{b}" for b in (25, 50) if b < n] + [f"n{nab}_ablation_D2"]
        for tag in tags:
            arms = ["A", "E", "B_ext", "C"] if tag == f"n{n}" else ["A", "C"] if tag in ("n25", "n50") else ["A", "B_ext", "C"]
            d = tmp_path / "outputs" / "exp8" / f"{tag}__{model}"
            d.mkdir(parents=True)
            with (d / "results.jsonl").open("w", encoding="utf-8") as fh:
                for arm in arms:
                    for cond in ("EN", "CS", "RU", "UR"):
                        for f in FAMS:
                            fh.write(json.dumps(rec(model, arm, cond, f, tag)) + "\n")
                            if int(f) <= 20:
                                fh.write(json.dumps(rec(model, arm, cond, f, tag, probe=True)) + "\n")
            (d / "run_manifest.json").write_text(json.dumps({"capability": {
                f"{model}/{a}": {"mmlu": 0.6 if a == "A" else 0.57, "urdummlu": 0.4} for a in arms}}))
    return tmp_path


def test_pooled_averages_labelled_seeds_only():
    pa = _load("phase2_analysis")
    got = pa.pooled([{"1": True, "2": None}, {"1": False, "2": None}, {"1": None, "3": True}])
    assert got == {"1": 0.5, "2": None, "3": 1.0}


def test_primary_finds_the_real_effect_and_not_the_null(fake_root, monkeypatch):
    pa = _load("phase2_analysis")
    monkeypatch.setattr(pa, "ROOT", fake_root)
    assert pa.main(["--judge-manifest", str(fake_root / "outputs/exp1/judge_validation_manifest.json")]) == 0
    out = json.loads((fake_root / "outputs/phase2_analysis/analysis.json").read_text(encoding="utf-8"))
    assert out["audit"].startswith("NOT_DONE") and out["ablation_domain"] == "D2"
    p = {t["model"]: t for t in out["primary"]}
    ll, ph = p["llama32"], p["phi3"]
    assert abs(ll["diff"] - (0.1 - 0.5)) < 1e-9 and ll["holm_reject"] and ll["finding"]
    c = {"tp": 30, "fn": 5, "tn": 95, "fp": 10}   # one error cell for both arms: corrected = raw x (PPV - FOR)
    assert abs(ll["diff_corrected"] - ll["diff"] * (c["tp"] / 40 - c["fn"] / 100)) < 1e-9
    assert ll["diff_corrected_ci_hi"] < 0
    assert ph["diff"] == 0 and ph["p_perm"] == 1 and not ph["finding"]
    assert set(ll["per_seed_mcnemar"]) == {"n60", "n60_s43", "n60_s44"}
    rows = list(csv.DictReader((fake_root / "outputs/phase2_analysis/comparisons.csv").open(encoding="utf-8")))
    kinds = collections.Counter((r["model"], r["kind"]) for r in rows)
    assert kinds[("llama32", "ncurve")] == 6 and kinds[("phi3", "ncurve")] == 4      # budgets 25, 50, N x CS/EN
    ab = [r for r in rows if r["kind"] == "ablation" and r["model"] == "llama32" and r["condition"] == "CS"]
    assert {r["n_families"] for r in ab} == {"30"}                                   # only the withheld domain
    cap = {(r["model"], r["arm"]): r for r in out["capability"]}
    assert abs(cap[("phi3", "C")]["mmlu_retention"] - 0.95) < 1e-9


def test_audit_make_is_blinded_balanced_and_apply_switches_by_fisher(fake_root, monkeypatch):
    ea = _load("exp8_audit")
    monkeypatch.setattr(ea, "ROOT", fake_root)
    monkeypatch.setattr(ea, "OUT", fake_root / "outputs/exp8/audit")
    monkeypatch.setattr(ea, "prompt_text", lambda: collections.defaultdict(lambda: "a prompt"))
    assert ea.cmd_make() == 0
    assert ea.cmd_make() == 3                                                        # never overwrites a sent audit
    out = fake_root / "outputs/exp8/audit"
    text = (out / "audit_file.csv").read_text(encoding="utf-8")
    assert not any(w in text for w in ("phi3", "llama32", "B_ext", "n39", "True", "False"))
    key = list(csv.DictReader((out / "audit_key.csv").open(encoding="utf-8")))
    cell = collections.Counter((k["task"], k["model"], k["condition"], k["judge_flag"]) for k in key)
    # Llama CS: plenty flagged in every arm -> 30 = 10 per arm; Phi-3 CS: few flagged in A (topped up)
    assert cell[("harm", "llama32", "CS", "True")] == 30 and cell[("harm", "phi3", "EN", "False")] == 30
    per_arm = collections.Counter(k["arm"] for k in key if (k["task"], k["model"], k["condition"], k["judge_flag"])
                                  == ("harm", "llama32", "CS", "True"))
    assert per_arm == {"A": 10, "B_ext": 10, "C": 10}
    short = collections.Counter(k["arm"] for k in key if (k["task"], k["model"], k["condition"], k["judge_flag"])
                                == ("harm", "phi3", "CS", "True"))
    assert short["A"] == 3 and sum(short.values()) == 30 and abs(short["C"] - short["B_ext"]) <= 1
    assert sum(n for (t, m, c, _), n in cell.items() if t == "harmless" and m == "phi3" and c == "RU") == 15
    assert len({(k["row_id"], k["response_sha256"]) for k in key}) == len(key)        # no response twice
    # reviewer: Phi-3 CS flagged responses half not harmful (Phase 1: 14/14) -> switch; others agree
    with (out / "returned.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["review_id", "label", "notes", "reviewer_id"])
        for i, k in enumerate(key):
            flag = k["judge_flag"] == "True"
            lab = ("no" if (k["model"], k["condition"]) == ("phi3", "CS") and i % 2 else "yes") if flag else "no"
            w.writerow([k["review_id"], lab, "", ""])
    assert ea.cmd_apply(out / "returned.csv", None, fake_root / "outputs/exp1/judge_validation_manifest.json") == 1
    assert ea.cmd_apply(out / "returned.csv", "UU", fake_root / "outputs/exp1/judge_validation_manifest.json") == 0
    res = json.loads((out / "audit_result.json").read_text(encoding="utf-8"))
    assert res["cells"]["phi3|CS"]["switch"] and not res["cells"]["llama32|EN"]["switch"]
    assert res["reviewer_ids"] == ["UU"]
    pa = _load("phase2_analysis")                                                    # the analysis picks it up
    monkeypatch.setattr(pa, "ROOT", fake_root)
    assert pa.main(["--judge-manifest", str(fake_root / "outputs/exp1/judge_validation_manifest.json")]) == 0
    a = json.loads((fake_root / "outputs/phase2_analysis/analysis.json").read_text(encoding="utf-8"))
    assert a["audit"] == "APPLIED" and a["error_cell_source"]["phi3|CS"] == "audit"
    assert a["error_cell_source"]["llama32|EN"] == "phase1"
    shutil.rmtree(out)
