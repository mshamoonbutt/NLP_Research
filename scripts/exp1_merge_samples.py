#!/usr/bin/env python3
"""Merge annotated Exp 1 samples (e.g. a validation sample + its top-up) into
one folder for scripts/calibrate_judge.py, which reads a single folder.

    python scripts/exp1_merge_samples.py \
        --sample-dirs outputs/exp1/validation-kaggle-01 outputs/exp1/validation-kaggle-01-topup1 \
        --out-dir outputs/exp1/validation-kaggle-01-merged

Concatenates items.csv, rater1.csv, rater2.csv and adjudication.csv. Refuses
samples with a different role, release or split, or that share a family or
an item. The merged sample_manifest.json lists every source with its item
hash; agreement is computed by calibrate_judge.py over all merged items.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from csjail.artifacts import sha256_json  # noqa: E402
from csjail.judge_validation import read_csv  # noqa: E402
from csjail.utils.io import write_text_lf  # noqa: E402

FILES = ("items.csv", "rater1.csv", "rater2.csv", "adjudication.csv")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-dirs", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--role", choices=["validation", "development"], default=None,
                    help="role of the merged sample (default: the sources' role). "
                         "'development' turns an annotated sample into a rubric-iteration set: "
                         "calibrate_judge.py then writes feedback reports, never a manifest.")
    args = ap.parse_args(argv)
    dirs = [Path(d) for d in args.sample_dirs]
    mans = [json.loads((d / "sample_manifest.json").read_text(encoding="utf-8")) for d in dirs]
    for key in ("role", "dataset_version", "split_id"):
        vals = {m.get(key) for m in mans}
        if len(vals) != 1:
            print(f"FAIL: samples differ in {key}: {vals}", file=sys.stderr)
            return 1
    fams = [f for m in mans for f in m["families"]]
    if len(fams) != len(set(fams)):
        print("FAIL: samples share families; a top-up must use new families", file=sys.stderr)
        return 1

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    merged = {}
    for name in FILES:
        rows, fields = [], []
        for d in dirs:
            with (d / name).open("r", encoding="utf-8-sig", newline="") as f:
                r = csv.DictReader(f)
                fields += [c for c in (r.fieldnames or []) if c not in fields]
                rows += list(r)
        with (out / name).open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, restval="")
            w.writeheader()
            w.writerows(rows)
        merged[name] = rows
    ids = [r["item_id"] for r in merged["items.csv"]]
    if len(ids) != len(set(ids)):
        print("FAIL: samples share item IDs", file=sys.stderr)
        return 1

    items = read_csv(out / "items.csv")
    write_text_lf(out / "sample_manifest.json", json.dumps({
        "kind": "exp1_sample_manifest_merged",
        "role": args.role or mans[0]["role"], "sample_kind": "representative",
        "source_role": mans[0]["role"],
        "dataset_version": mans[0]["dataset_version"], "split_id": mans[0]["split_id"],
        "seed": [m.get("seed") for m in mans],
        "models": sorted({x for m in mans for x in m.get("models", [])}),
        "n_items": len(items), "families": fams, "item_ids": ids,
        "items_sha256": sha256_json(items),
        "merged_from": [{"dir": str(d), "n_items": m.get("n_items"),
                         "items_sha256": m.get("items_sha256"),
                         "conditions": m.get("conditions"),
                         "responses_from_run": m.get("responses_from_run")}
                        for d, m in zip(dirs, mans)],
    }, ensure_ascii=False, indent=2))
    print(f"[merge] {len(items)} items from {len(dirs)} samples ({len(fams)} families) -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
