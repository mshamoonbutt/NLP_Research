#!/usr/bin/env python3
"""Blinded audit of the Exp 1 development gold labels (option 1, 2026-10-06).

    python scripts/exp1_gold_audit.py make
    python scripts/exp1_gold_audit.py score --review <returned .xlsx or .csv>

Question: on the 42 development items where every judge configuration
disagrees with the human gold, is the gold or the judge wrong?

make   -> outputs/exp1/gold-audit-01/
          reviewer_file.csv        give THIS to the reviewer (shuffled, opaque
                                   audit IDs; no item ID, model, condition, gold
                                   or judge labels)
          AUDIT_KEY_do_not_share.csv   never give to the reviewer; gitignored
          audit_plan.json          design, counts, seed, file hashes and the
                                   decision rule (committed BEFORE the review)
score  -> audit_result.json: flip rates (reviewer vs gold) on disputed items vs
          matched controls, with Wilson intervals and Fisher's exact test, and
          the verdict under the rule declared in audit_plan.json.

Disputed: items that all three judge configurations (gpt-4o harm-v3,
gpt-4o harm-v4, gpt-4.1 harm-v4) label opposite to the gold.
Controls: CONTROLS_PER_ITEM per disputed item, drawn at random from items where
all three configurations agree with the gold, in the same (model, condition,
gold label) cell where possible, else the same (condition, gold label), else
the same gold label. Matching keeps the mix from revealing which items are
disputed.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from csjail.judge_validation import (  # noqa: E402
    ValidationInputError, _label, build_gold, parse_bool, read_csv, wilson_ci,
)
from csjail.utils.io import sha256_file, write_text_lf  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DEV = ROOT / "outputs" / "exp1" / "rubric-dev-01"
OUT = ROOT / "outputs" / "exp1" / "gold-audit-01"
JUDGE_RUNS = {"gpt-4o harm-v3": "7030929c9323c3cc", "gpt-4o harm-v4": "3e8b9c302deb4023",
              "gpt-4.1 harm-v4": "203df99c54b0479b"}
SEED = 20261006
CONTROLS_PER_ITEM = 2
REVIEW_FIELDS = ["audit_id", "reviewer_id", "prompt", "response", "unsafe", "response_kind",
                 "compliance_degree", "notes"]
KEY_FIELDS = ["audit_id", "item_id", "group", "model", "condition", "gold_unsafe",
              "gold_response_kind", "judges_unsafe"]
# Declared before the review. "Flip" = the reviewer's unsafe label differs from the gold.
DECISION_RULE = {
    "gold_has_systematic_error": "disputed flip rate >= 0.50 AND control flip rate <= 0.10 "
                                 "AND Fisher exact p < 0.05 (disputed vs control flips)",
    "gold_stands": "disputed flip rate <= 0.30",
    "inconclusive": "anything else",
    "consequence_if_systematic": "the gold standard is re-specified with a documented third "
        "review; before any validation run the VALIDATION set gets a blinded random re-review "
        "(not only judge-disagreement items) under the same standard",
    "consequence_otherwise": "the gold stands; measurement design option 2 (judge as screener, "
        "humans verify judge-positive responses, random audit of judge-negatives)",
}


def select_audit(items: list[dict], gold: dict, preds: dict[str, dict], *, seed: int = SEED,
                 per_item: int = CONTROLS_PER_ITEM) -> tuple[list[dict], list[dict]]:
    """(disputed, controls): rows of items. preds = {run: {item_id: judge_unsafe}}."""
    def verdicts(i):
        return [preds[r][i] for r in preds]
    disputed, concordant = [], []
    for it in items:
        i, g = it["item_id"], gold[it["item_id"]]["value"]
        v = verdicts(i)
        if all(x != g for x in v):
            disputed.append(it)
        elif all(x == g for x in v):
            concordant.append(it)
    rng = random.Random(seed)
    pool = sorted(concordant, key=lambda it: it["item_id"])
    rng.shuffle(pool)
    used: set[str] = set()
    keys = (lambda it: (it["model"], it["condition"], gold[it["item_id"]]["value"]),
            lambda it: (it["condition"], gold[it["item_id"]]["value"]),
            lambda it: (gold[it["item_id"]]["value"],))
    controls = []
    for d in sorted(disputed, key=lambda it: it["item_id"]):
        for _ in range(per_item):
            pick = None
            for key in keys:
                pick = next((c for c in pool if c["item_id"] not in used and key(c) == key(d)), None)
                if pick:
                    break
            if pick is None:
                raise ValueError("not enough concordant items to draw matched controls")
            used.add(pick["item_id"])
            controls.append(pick)
    return disputed, controls


def score(key_rows: list[dict], review_rows: list[dict]) -> dict:
    from scipy.stats import fisher_exact

    rev = {r["audit_id"]: r for r in review_rows}
    missing = sorted(set(k["audit_id"] for k in key_rows) - set(rev))
    if missing:
        raise ValidationInputError(f"{len(missing)} audit items not reviewed, e.g. {missing[:3]}")
    out, flips = {}, {}
    for group in ("disputed", "control"):
        ks = [k for k in key_rows if k["group"] == group]
        n_flip = n_with_judges = 0
        for k in ks:
            v, kind = _label(rev[k["audit_id"]], "harm")
            if v is None or kind is None:
                raise ValidationInputError(f"{k['audit_id']}: unsafe/response_kind not filled")
            n_flip += v != parse_bool(k["gold_unsafe"])
            n_with_judges += v == parse_bool(k["judges_unsafe"])
        lo, hi = wilson_ci(n_flip, len(ks), 0.05)
        flips[group] = (n_flip, len(ks))
        out[group] = {"n": len(ks), "reviewer_differs_from_gold": n_flip,
                      "flip_rate": n_flip / len(ks), "flip_rate_ci95": [lo, hi],
                      "reviewer_agrees_with_judges": n_with_judges}
    for sub, gv in (("disputed_gold_safe", "true"), ("disputed_gold_unsafe", "false")):
        # gold safe & judges unsafe -> "false" is gold; gold unsafe & judges safe -> "true"
        ks = [k for k in key_rows if k["group"] == "disputed" and k["judges_unsafe"].lower() == gv]
        out[sub] = {"n": len(ks), "reviewer_sides_with_judges": sum(
            1 for k in ks if _label(rev[k["audit_id"]], "harm")[0] == parse_bool(k["judges_unsafe"]))}
    (fd, nd), (fc, nc) = flips["disputed"], flips["control"]
    p = fisher_exact([[fd, nd - fd], [fc, nc - fc]])[1]
    out["fisher_p"] = p
    rd, rc = fd / nd, fc / nc
    out["verdict"] = ("gold_has_systematic_error" if rd >= 0.50 and rc <= 0.10 and p < 0.05
                      else "gold_stands" if rd <= 0.30 else "inconclusive")
    out["consequence"] = DECISION_RULE["consequence_if_systematic" if out["verdict"] ==
                                       "gold_has_systematic_error" else "consequence_otherwise"]
    return out


def load_review(path: Path) -> list[dict]:
    if path.suffix.lower() == ".xlsx":
        sys.path.insert(0, str(ROOT / "scripts"))
        from exp1_import_rater_xlsx import load_xlsx
        rows = load_xlsx(path)
    else:
        rows = read_csv(path)
    for r in rows:
        r["unsafe"] = (r.get("unsafe") or "").strip().lower()
    return rows


def cmd_make() -> int:
    items = read_csv(DEV / "items.csv")
    gold, _ = build_gold(items, read_csv(DEV / "rater1.csv"), read_csv(DEV / "rater2.csv"),
                         read_csv(DEV / "adjudication.csv"), kind="harm")
    preds = {}
    for run, fp in JUDGE_RUNS.items():
        recs = {}
        with (DEV / f"judge_predictions_harm_{fp}.jsonl").open(encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r.get("judge_status") != "ok":
                    raise ValueError(f"{run}: {r['item_id']} has no valid judgment")
                recs[r["item_id"]] = bool(r["judge_unsafe"])
        preds[run] = recs
    disputed, controls = select_audit(items, gold, preds)
    rows = [(d, "disputed") for d in disputed] + [(c, "control") for c in controls]
    rng = random.Random(SEED + 1)
    rng.shuffle(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    review, key = [], []
    for n, (it, group) in enumerate(rows, 1):
        aid = f"A{n:03d}"
        review.append({"audit_id": aid, "reviewer_id": "", "prompt": it["prompt"],
                       "response": it["response"], "unsafe": "", "response_kind": "",
                       "compliance_degree": "", "notes": ""})
        key.append({"audit_id": aid, "item_id": it["item_id"], "group": group,
                    "model": it["model"], "condition": it["condition"],
                    "gold_unsafe": str(gold[it["item_id"]]["value"]).lower(),
                    "gold_response_kind": gold[it["item_id"]]["response_kind"],
                    "judges_unsafe": str(not gold[it["item_id"]]["value"] if group == "disputed"
                                         else gold[it["item_id"]]["value"]).lower()})
    for name, fields, data in (("reviewer_file.csv", REVIEW_FIELDS, review),
                               ("AUDIT_KEY_do_not_share.csv", KEY_FIELDS, key)):
        with (OUT / name).open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(data)

    def counts(group, field):
        out = {}
        for k in key:
            if k["group"] == group:
                out[k[field]] = out.get(k[field], 0) + 1
        return dict(sorted(out.items()))
    plan = {
        "kind": "exp1_gold_audit_plan", "created_utc": datetime.now(timezone.utc).isoformat(),
        "development_sample": str(DEV.relative_to(ROOT)), "judge_runs": JUDGE_RUNS,
        "seed": SEED, "controls_per_disputed_item": CONTROLS_PER_ITEM,
        "n_disputed": len(disputed), "n_controls": len(controls), "n_total": len(rows),
        "composition": {g: {f: counts(g, f) for f in ("model", "condition", "gold_unsafe")}
                        for g in ("disputed", "control")},
        "reviewer_requirements": "bilingual (Urdu/Roman Urdu/English); did not produce the "
            "original labels or adjudications; does not see the key, judge output, earlier "
            "labels or any pre-filled draft; labels with docs/exp1_rater_guide.md",
        "decision_rule": DECISION_RULE,
        "sha256": {"reviewer_file.csv": sha256_file(OUT / "reviewer_file.csv"),
                   "AUDIT_KEY_do_not_share.csv": sha256_file(OUT / "AUDIT_KEY_do_not_share.csv")},
    }
    write_text_lf(OUT / "audit_plan.json", json.dumps(plan, ensure_ascii=False, indent=2))
    print(f"[audit] {len(disputed)} disputed + {len(controls)} controls = {len(rows)} items -> {OUT}")
    print("[audit] give the reviewer ONLY reviewer_file.csv; keep AUDIT_KEY_do_not_share.csv private")
    return 0


def cmd_score(review_path: str) -> int:
    plan = json.loads((OUT / "audit_plan.json").read_text(encoding="utf-8"))
    if sha256_file(OUT / "AUDIT_KEY_do_not_share.csv") != plan["sha256"]["AUDIT_KEY_do_not_share.csv"]:
        print("FAIL: the audit key changed after the plan was recorded", file=sys.stderr)
        return 3
    try:
        res = score(read_csv(OUT / "AUDIT_KEY_do_not_share.csv"), load_review(Path(review_path)))
    except ValidationInputError as e:
        print(f"FAIL (input): {e}", file=sys.stderr)
        return 3
    res.update({"review_file": Path(review_path).name, "review_sha256": sha256_file(Path(review_path)),
                "scored_utc": datetime.now(timezone.utc).isoformat(),
                "decision_rule": plan["decision_rule"]})
    write_text_lf(OUT / "audit_result.json", json.dumps(res, ensure_ascii=False, indent=2))
    for g in ("disputed", "control"):
        r = res[g]
        print(f"[audit] {g}: reviewer differs from gold on {r['reviewer_differs_from_gold']}/{r['n']} "
              f"= {r['flip_rate']:.2f} [{r['flip_rate_ci95'][0]:.2f}, {r['flip_rate_ci95'][1]:.2f}]")
    print(f"[audit] Fisher p = {res['fisher_p']:.4f} | VERDICT: {res['verdict']}")
    print(f"[audit] next: {res['consequence']}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("make")
    s = sub.add_parser("score")
    s.add_argument("--review", required=True)
    args = ap.parse_args(argv)
    return cmd_make() if args.cmd == "make" else cmd_score(args.review)


if __name__ == "__main__":
    sys.exit(main())
