"""Exp 6 pair review: blinded file -> verified pair sets with equal budgets."""
from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

from csjail.prefdata import read_pairs, write_pairs

ROOT = Path(__file__).resolve().parent.parent


def _load():
    spec = importlib.util.spec_from_file_location("exp6_review", ROOT / "scripts" / "exp6_review.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_review_round_trip(tmp_path):
    rv = _load()
    exp6 = tmp_path / "exp6"
    doms = {"phi3": ["D1", "D6", "D2"], "llama32": ["D1", "D2", "D6", "D3"]}
    for model, ds in doms.items():
        d = exp6 / model
        d.mkdir(parents=True)
        write_pairs(str(d / "pairs_cs_all.jsonl"), [
            {"base_id": f"{model}-{i}", "domain_id": dm, "order_rank": i, "prompt": "p", "rejected": "r",
             "chosen": "c"} for i, dm in enumerate(ds)])
        (d / "pairs_manifest.json").write_text(json.dumps({"model": model, "split_id": "main-split"}))
    ab = tmp_path / "ab.json"
    ab.write_text(json.dumps({"meta": {"ablation_domain": "D6", "split_id": "ab-split"}}))
    assert rv.main(["make", "--exp6-root", str(exp6)]) == 0
    assert rv.main(["make", "--exp6-root", str(exp6)]) == 3          # never overwrites a sent review
    with (exp6 / "review" / "review_file.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 7 and "model" not in rows[0]                  # blinded
    key = {r["review_id"]: r for r in csv.DictReader((exp6 / "review" / "review_key.csv").open(encoding="utf-8"))}
    for r in rows:   # drop phi3-0 (not harmful) and llama32-1 (bad refusal)
        b = key[r["review_id"]]["base_id"]
        r.update(rejected_harmful="no" if b == "phi3-0" else "yes",
                 chosen_refusal_ok="no" if b == "llama32-1" else "yes", chosen_natural="4", reviewer_id="UU")
    ret = exp6 / "review" / "review_returned.csv"
    with ret.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    assert rv.main(["apply", "--exp6-root", str(exp6), "--review", str(ret), "--ablation-split", str(ab)]) == 0
    s = json.loads((exp6 / "review" / "verified_summary.json").read_text())
    assert s["models"]["phi3"]["kept"] == 2 and s["models"]["llama32"]["kept"] == 3
    assert s["budget_main"] == {"phi3": 2, "llama32": 3}             # per model
    assert s["budget_ablation"] == {"phi3": 1, "llama32": 2}         # phi3 keeps D6 + D2; minus D6 -> 1
    assert [p["base_id"] for p in read_pairs(str(exp6 / "phi3_verified" / "pairs_cs_all.jsonl"))] == ["phi3-1", "phi3-2"]
    assert [p["domain_id"] for p in read_pairs(str(exp6 / "llama32_ablation_D6" / "pairs_cs_all.jsonl"))] == ["D1", "D3"]
    man = json.loads((exp6 / "llama32_ablation_D6" / "pairs_manifest.json").read_text())
    assert man["split_id"] == "ab-split" and man["excluded_domains"] == ["D6"]
    rows[0]["reviewer_id"] = ""                                        # every row needs the reviewer
    with ret.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    assert rv.main(["apply", "--exp6-root", str(exp6), "--review", str(ret), "--ablation-split", str(ab)]) == 1
    assert rv.main(["apply", "--exp6-root", str(exp6), "--review", str(ret), "--ablation-split", str(ab),
                    "--reviewer-id", "UU", "--provenance", "test"]) == 0     # fills the blank only
    s = json.loads((exp6 / "review" / "verified_summary.json").read_text())
    assert s["reviewer_id_filled_from_cli"] == 1 and s["provenance"] == "test"
