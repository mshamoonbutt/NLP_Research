"""Option 2 (screener + human verification): blinded sample -> labels -> ASR round trip."""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load():
    spec = importlib.util.spec_from_file_location("exp2_verify", ROOT / "scripts" / "exp2_verify.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["exp2_verify"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_make_score_round_trip(tmp_path, monkeypatch):
    v = load()
    from csjail.pipeline import judgment_key
    fp = {"fingerprint_id": "screener"}
    monkeypatch.setattr(v, "screener_fingerprint", lambda: fp)
    monkeypatch.setattr(v, "prompts_by_row", lambda: {f"{n}::EN": "p" for n in range(200)})
    run = tmp_path / "run"
    run.mkdir()
    gens, truth, judg = [], {}, []
    for n in range(200):                      # one stratum, 200 responses
        g = {"kind": "generation", "gen_key": f"g{n}", "row_id": f"{n}::EN", "response": f"r{n}",
             "response_sha256": f"h{n}", "model": "m", "condition": "EN", "split": "eval_main",
             "generation_status": "ok"}
        gens.append(g)
        truth[g["gen_key"]] = n < 20 or n in (150, 160)        # 22 unsafe; 2 the screener misses
        flagged = n < 30                                       # 20 true + 10 false positives
        st = "parse_error" if n == 29 else "ok"               # an unparsed one is reviewed too
        judg.append({"judge_key": judgment_key(g, fp), "judge_status": st, "judge_unsafe": flagged})
    for name, recs in (("generations.jsonl", gens), ("judgments.jsonl", judg)):
        (run / name).write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
    out = tmp_path / "verify"
    assert v.main(["make", "--run-dir", str(run), "--out", str(out)]) == 0
    plan = json.loads((out / "audit_plan.json").read_text(encoding="utf-8"))
    assert plan["counts"]["m|EN"] == {"n_pos": 30, "n_neg": 170, "n_neg_sampled": 17}
    review = list(csv.DictReader((out / "reviewer_file.csv").open(encoding="utf-8")))
    assert len(review) == 47 and not {"model", "condition", "item_id"} & set(review[0])   # blinded
    key = {k["audit_id"]: k["item_id"] for k in csv.DictReader(
        (out / "AUDIT_KEY_do_not_share.csv").open(encoding="utf-8"))}
    for r in review:
        u = truth[key[r["audit_id"]]]
        r.update(reviewer_id="AB", unsafe=str(u).lower(),
                 response_kind="harmful_compliance" if u else "refusal")
    halves = [tmp_path / "a.csv", tmp_path / "b.csv"]         # two reviewers split the file
    for path, part in zip(halves, (review[:20], review[20:])):
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(review[0]))
            w.writeheader()
            w.writerows(part)
    assert v.main(["score", "--out", str(out), "--review", *map(str, halves)]) == 0
    res = json.loads((out / "verification_result.json").read_text(encoding="utf-8"))["per_stratum"]["m|EN"]
    audited_unsafe = sum(truth[key[r["audit_id"]]] for r in review
                         if int(key[r["audit_id"]][1:]) >= 30)
    assert res["unsafe_flagged"] == 20 and res["unsafe_audited"] == audited_unsafe
    assert abs(res["asr"] - (20 + audited_unsafe * 10) / 200) < 1e-9       # weight 170/17 = 10
    assert res["asr_interval"][0] <= res["asr"] <= res["asr_interval"][1]
    assert res["asr_interval"][0] >= 0.10                                  # flagged part is a census
    assert abs(res["screener_precision"] - 20 / 30) < 1e-9
    review[0]["reviewer_id"] = ""
    with halves[0].open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(review[0]))
        w.writeheader()
        w.writerows(review[:20])
    assert v.main(["score", "--out", str(out), "--review", *map(str, halves)]) == 3   # reviewer_id required
