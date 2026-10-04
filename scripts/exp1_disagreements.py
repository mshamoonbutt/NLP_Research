#!/usr/bin/env python3
"""List Exp 1 rater disagreements for adjudication (CPU; no judge involved).

    python scripts/exp1_disagreements.py --sample-dir outputs/exp1/validation-kaggle-01
    python scripts/exp1_disagreements.py --kind benign --sample-dir outputs/exp1/benign-validation-kaggle-01

Writes adjudication_todo.csv: every item where the two raters differ in the
binary label or response_kind, both raters' labels side by side, gold_*
columns blank. The adjudicator fills gold_*, resolution and adjudicator, and
saves those rows as adjudication.csv. Items a rater left unlabeled are listed
separately: they go back to that rater, not to adjudication (exit code 2).
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from csjail.judge_validation import _label, read_csv  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-dir", required=True)
    ap.add_argument("--kind", choices=["harm", "benign"], default="harm")
    args = ap.parse_args(argv)
    sd = Path(args.sample_dir)
    key = "refused" if args.kind == "benign" else "unsafe"
    extra = [] if args.kind == "benign" else ["compliance_degree"]
    labels = [key, "response_kind", *extra]

    items = read_csv(sd / "items.csv")
    r1 = {r["item_id"]: r for r in read_csv(sd / "rater1.csv")}
    r2 = {r["item_id"]: r for r in read_csv(sd / "rater2.csv")}
    fields = (["item_id", "model", "condition", "prompt", "response"]
              + [f"rater{n}_{f}" for n in (1, 2) for f in labels]
              + [f"gold_{f}" for f in labels] + ["resolution", "adjudicator", "notes"])

    todo, unlabeled = [], {"rater1": [], "rater2": []}
    for it in items:
        iid = it["item_id"]
        a, b = r1.get(iid, {}), r2.get(iid, {})
        la, lb = _label(a, args.kind), _label(b, args.kind)
        if None in la or None in lb:
            if None in la:
                unlabeled["rater1"].append(iid)
            if None in lb:
                unlabeled["rater2"].append(iid)
            continue
        if la != lb:
            row = {k: it.get(k, "") for k in ("item_id", "model", "condition", "prompt", "response")}
            for n, r in ((1, a), (2, b)):
                row.update({f"rater{n}_{f}": r.get(f, "") for f in labels})
            todo.append(row)

    out = sd / "adjudication_todo.csv"
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(todo)
    n_ok = len(items) - len(set(unlabeled["rater1"]) | set(unlabeled["rater2"]))
    print(f"[disagree] {len(todo)} disagreements among {n_ok} double-labelled items "
          f"-> {out}")
    if unlabeled["rater1"] or unlabeled["rater2"]:
        for name, ids in unlabeled.items():
            if ids:
                print(f"[disagree] {name} left {len(ids)} items unlabeled, e.g. {ids[:5]}",
                      file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
