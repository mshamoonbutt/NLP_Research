#!/usr/bin/env python3
"""Build the Exp 1 DEVELOPMENT gold file from the two prefilled label files.

    python scripts/exp1_dev_gold.py --sample-dir outputs/exp1/development-cpu-20260926

Reads (under <sample-dir>/annotations/): the two original label files
(rater1_*.csv, rater2_*.csv, never modified), adjudication_questions.csv
(human decisions, may be blank) and verification_metadata.json (human
verification fields, preserved across runs). Writes merged_labels.csv
(merged by item_id; no prompt/response text), refreshes the derived part of
verification_metadata.json, and writes <sample-dir>/development_gold.csv.

gold_status per item:
  human_adjudicated     a human filled gold_* + adjudicator on the sheet
  pending_adjudication  on the sheet, no human decision yet (prefill kept)
  pending_disagreement  the two files disagree and no decision exists
  ai_prefill            not flagged; AI prefill, human verification not recorded
Re-run after every round of decisions. Never overwrites the originals.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from csjail.judge_validation import parse_bool  # noqa: E402
from csjail.outcomes import COMPLIANCE_DEGREES, RESPONSE_KINDS  # noqa: E402
from csjail.utils.io import sha256_file, write_text_lf  # noqa: E402

LABELS = ("unsafe", "response_kind", "compliance_degree")


def read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def consistent(unsafe, kind, degree) -> bool:
    if unsafe is None or kind not in RESPONSE_KINDS or degree not in COMPLIANCE_DEGREES:
        return False
    return (kind == "harmful_compliance" and degree in ("full", "partial")) if unsafe \
        else (kind != "harmful_compliance" and degree == "none")


def build(sample_dir: Path) -> dict:
    a = sample_dir / "annotations"
    [f1] = sorted(a.glob("rater1_*.csv"))
    [f2] = sorted(a.glob("rater2_*.csv"))
    r1 = {r["item_id"]: r for r in read(f1)}
    r2 = {r["item_id"]: r for r in read(f2)}
    items = {r["item_id"]: r for r in read(sample_dir / "items.csv")}
    if set(r1) != set(items) or set(r2) != set(items):
        raise ValueError("label files and items.csv cover different item_ids")
    decisions = {r["item_id"]: r for r in read(a / "adjudication_questions.csv")}

    merged, gold = [], []
    for iid in sorted(items):
        it, x, y = items[iid], r1[iid], r2[iid]
        same = all(x[k] == y[k] for k in LABELS)
        meta = {"item_id": iid, "model": it["model"], "condition": it["condition"],
                "base_id": it["base_id"], "domain_id": it["domain_id"]}
        merged.append({**meta, **{f"r1_{k}": x[k] for k in ("rater_id", *LABELS, "notes")},
                       **{f"r2_{k}": y[k] for k in ("rater_id", *LABELS, "notes")},
                       "labels_identical": same})
        d = decisions.get(iid, {})
        dec = (parse_bool(d.get("gold_unsafe")), (d.get("gold_response_kind") or "").strip(),
               (d.get("gold_compliance_degree") or "").strip())
        decided = all(v not in (None, "") for v in dec) and (d.get("adjudicator") or "").strip()
        if decided:
            if not consistent(*dec):
                raise ValueError(f"{iid}: inconsistent adjudicated labels {dec}")
            lab, status, src = dec, "human_adjudicated", "human adjudication"
        else:
            lab = (parse_bool(x["unsafe"]), x["response_kind"], x["compliance_degree"])
            status = ("pending_adjudication" if iid in decisions else
                      "ai_prefill" if same else "pending_disagreement")
            src = "AI prefill"
        gold.append({**meta, "gold_unsafe": lab[0], "gold_response_kind": lab[1],
                     "gold_compliance_degree": lab[2], "gold_status": status,
                     "label_source": src, "adjudicator": (d.get("adjudicator") or "").strip(),
                     "decision_rule": (d.get("decision_rule") or "").strip()})
    write(a / "merged_labels.csv", merged)
    write(sample_dir / "development_gold.csv", gold)

    mpath = a / "verification_metadata.json"
    meta = json.loads(mpath.read_text(encoding="utf-8")) if mpath.exists() else {
        "human_verification": {
            "labels_prefilled_by_ai": True,
            "human_verifier_ids": None,      # anonymized IDs, e.g. ["V1", "V2"]; not recorded yet
            "verification_date": None,       # not recorded yet
            "what_was_verified": None,       # e.g. "all 160 items" / "flagged items only"
            "note": "Fill from real records only; leave null if unknown."}}
    meta["derived"] = {
        "source_files": {f.name: sha256_file(f) for f in (f1, f2)},
        "rater_id_values": sorted({r["rater_id"] for r in (*r1.values(), *r2.values())}),
        "n_items": len(items),
        "n_items_labels_identical": sum(m["labels_identical"] for m in merged),
        "independent_annotations": False if all(m["labels_identical"] for m in merged) else None,
        "gold_status_counts": dict(Counter(g["gold_status"] for g in gold)),
        "unsafe_by_condition": dict(Counter(g["condition"] for g in gold if g["gold_unsafe"])),
    }
    write_text_lf(mpath, json.dumps(meta, indent=2) + "\n")
    return meta["derived"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-dir", required=True)
    args = ap.parse_args(argv)
    print(json.dumps(build(Path(args.sample_dir)), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
