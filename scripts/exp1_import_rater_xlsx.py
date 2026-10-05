#!/usr/bin/env python3
"""Import returned rater spreadsheets (.xlsx) into an Exp 1 sample folder.

    python scripts/exp1_import_rater_xlsx.py --sample-dir outputs/exp1/validation-kaggle-01 \
        --rater1 rater1_annotated.xlsx --rater2 rater2_annotated.xlsx
    python scripts/exp1_import_rater_xlsx.py --kind benign --sample-dir ... --rater1 ... --rater2 ...

Validates each workbook against items.csv (same item IDs, each exactly once;
valid labels; a rater_id on every row; different IDs in the two files), then
writes rater1.csv / rater2.csv in the CSV format calibrate_judge.py reads.
Normalized: Excel booleans -> true/false, whole-number floats in rater_id
(1.0 -> 1), empty cells -> "". Labels are never changed. The original
workbooks are copied byte-for-byte to <sample-dir>/annotations/ and their
SHA-256 recorded in annotation_import.json. Requires openpyxl.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from csjail.judge_validation import ValidationInputError, _label, read_csv  # noqa: E402
from csjail.utils.io import sha256_file, write_text_lf  # noqa: E402

FIELDS = {"harm": ["item_id", "rater_id", "prompt", "response", "unsafe", "response_kind",
                   "compliance_degree", "notes"],
          "benign": ["item_id", "rater_id", "prompt", "response", "refused", "response_kind", "notes"]}


def cell(v) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v).strip() if not isinstance(v, str) else v


def load_xlsx(path: Path) -> list[dict]:
    import openpyxl

    ws = openpyxl.load_workbook(path, read_only=True, data_only=True).worksheets[0]
    rows = list(ws.iter_rows(values_only=True))
    hdr = [str(h).strip() if h is not None else "" for h in rows[0]]
    # Excel omits trailing empty cells (e.g. blank notes), so short rows are padded.
    return [{h: cell(r[i]) if i < len(r) else "" for i, h in enumerate(hdr)} for r in rows[1:]
            if any(v not in (None, "") for v in r)]


def check(rows: list[dict], items: dict, kind: str, name: str) -> list[dict]:
    key = "refused" if kind == "benign" else "unsafe"
    missing_cols = [f for f in FIELDS[kind] if f not in rows[0]]
    if missing_cols:
        raise ValidationInputError(f"{name}: missing columns {missing_cols}")
    ids = Counter(r["item_id"] for r in rows)
    dup = [i for i, c in ids.items() if c > 1]
    if dup or set(ids) != set(items):
        raise ValidationInputError(
            f"{name}: item IDs differ from items.csv (duplicates {dup[:3]}, missing "
            f"{sorted(set(items) - set(ids))[:3]}, unknown {sorted(set(ids) - set(items))[:3]})")
    for r in rows:
        r[key] = r[key].strip().lower()
        if not r["rater_id"].strip():
            raise ValidationInputError(f"{name}: empty rater_id on {r['item_id']}")
        _label(r, kind)                     # raises on an invalid response_kind
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-dir", required=True)
    ap.add_argument("--kind", choices=["harm", "benign"], default="harm")
    ap.add_argument("--rater1", required=True, help="rater 1 workbook (.xlsx)")
    ap.add_argument("--rater2", required=True, help="rater 2 workbook (.xlsx)")
    args = ap.parse_args(argv)
    sd = Path(args.sample_dir)
    items = {r["item_id"]: r for r in read_csv(sd / "items.csv")}
    try:
        r1 = check(load_xlsx(Path(args.rater1)), items, args.kind, "rater1")
        r2 = check(load_xlsx(Path(args.rater2)), items, args.kind, "rater2")
        ids1, ids2 = {r["rater_id"] for r in r1}, {r["rater_id"] for r in r2}
        if ids1 & ids2:
            raise ValidationInputError(f"the same rater_id appears in both files: {ids1 & ids2}")
    except ValidationInputError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1

    ann = sd / "annotations"
    ann.mkdir(exist_ok=True)
    sources = {}
    for name, rows, src in (("rater1", r1, Path(args.rater1)), ("rater2", r2, Path(args.rater2))):
        with (sd / f"{name}.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS[args.kind], extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        dst = ann / src.name
        shutil.copyfile(src, dst)
        sources[name] = {"workbook": dst.name, "sha256": sha256_file(dst), "n_rows": len(rows),
                         "rater_ids": sorted({r["rater_id"] for r in rows}),
                         "csv_sha256": sha256_file(sd / f"{name}.csv")}
        print(f"[import] {name}: {len(rows)} rows -> {sd / f'{name}.csv'}")
    write_text_lf(sd / "annotation_import.json", json.dumps({
        "kind": f"exp1_annotation_import_{args.kind}",
        "imported_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "sources": sources,
        "normalization": "Excel booleans -> true/false; whole-number rater_id floats -> int; "
                         "labels unchanged; prompt/response kept as returned (display copies; "
                         "gold is keyed by item_id)",
        "annotation_process": {"annotators": None, "dates": None, "tools_or_assistance": None,
                               "note": "fill from real records; reported in the paper as performed"},
    }, ensure_ascii=False, indent=2))
    print(f"[import] originals + annotation_import.json -> {ann}, {sd}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
