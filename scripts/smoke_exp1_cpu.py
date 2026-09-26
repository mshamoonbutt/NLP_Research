#!/usr/bin/env python3
"""CPU smoke of the Exp 1 validation logic -- no model, no API.

Builds a synthetic sample (2 stub models x 4 conditions x N items), two
independent stub raters with a few disagreements plus adjudication, and runs
csjail.judge_validation.evaluate against several stub judges:

  good judge                 -> PASS
  false positives on RU      -> FAIL (precision)
  every output unparseable   -> INSUFFICIENT_EVIDENCE (the reproduced bug: this
                                used to print "OK: min_precision N/A")
  EN condition missing       -> INSUFFICIENT_EVIDENCE
  one disagreement unadjudicated -> INSUFFICIENT_EVIDENCE

    python scripts/smoke_exp1_cpu.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from csjail.judge import load_judge_config  # noqa: E402
from csjail.judge_validation import FAIL, INSUFFICIENT, PASS, build_gold, evaluate  # noqa: E402

N = 30


def make_sample():
    items, r1, r2, adj = [], [], [], []
    for model in ("mA", "mB"):
        for cond in ("CS", "EN", "RU", "UR"):
            for i in range(N):
                iid = f"{model}:{cond}:{i}"
                unsafe = i % 2 == 0
                kind = "harmful_compliance" if unsafe else "refusal"
                items.append({"item_id": iid, "model": model, "condition": cond,
                              "sample_kind": "representative", "prompt": "p", "response": "r"})
                r1.append({"item_id": iid, "rater_id": "R1", "unsafe": str(unsafe),
                           "response_kind": kind})
                flip = i == 3  # a disagreement, adjudicated below
                r2.append({"item_id": iid, "rater_id": "R2", "unsafe": str(not unsafe if flip else unsafe),
                           "response_kind": ("harmful_compliance" if flip else kind)})
                if flip:
                    adj.append({"item_id": iid, "gold_unsafe": str(unsafe),
                                "gold_response_kind": kind, "resolution": "discussed",
                                "adjudicator": "A1"})
    return items, r1, r2, adj


def preds_from(items, gold, fn):
    return {it["item_id"]: fn(it, gold[it["item_id"]]) for it in items}


def main() -> int:
    cfg = load_judge_config().validation
    items, r1, r2, adj = make_sample()
    gold, grep = build_gold(items, r1, r2, adj)
    ok = True

    def check(name, expected, preds, its=items, gr=grep, gd=gold):
        nonlocal ok
        res = evaluate(its, gd, preds, cfg, gold_report=gr)
        good = res["status"] == expected
        ok &= good
        print(f"[{'OK' if good else 'FAIL'}] {name}: {res['status']} (expected {expected}) "
              f"{ {c: s['status'] for c, s in res['condition_status'].items()} }")

    perfect = lambda it, g: {"status": "ok", "value": g["value"], "response_kind": g["response_kind"]}  # noqa: E731
    check("good judge", PASS, preds_from(items, gold, perfect))
    check("RU false positives", FAIL, preds_from(items, gold, lambda it, g: {
        "status": "ok", "value": True if it["condition"] == "RU" else g["value"],
        "response_kind": "harmful_compliance"}))
    check("all unparseable", INSUFFICIENT, preds_from(items, gold, lambda it, g: {
        "status": "parse_error", "value": None, "response_kind": None}))
    no_en = [it for it in items if it["condition"] != "EN"]
    check("EN missing", INSUFFICIENT, preds_from(no_en, gold, perfect), its=no_en)
    gold2, grep2 = build_gold(items, r1, r2, adj[1:])
    check("unadjudicated disagreement", INSUFFICIENT, preds_from(
        [it for it in items if it["item_id"] in gold2], gold2, perfect), gr=grep2, gd=gold2)
    print(f"pre-adjudication agreement: {grep['pre_adjudication_agreement']}")
    print("SMOKE OK" if ok else "SMOKE FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
