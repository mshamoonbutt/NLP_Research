#!/usr/bin/env python3
"""Over-refusal probe v2 (Exp 8): the team's harmless prompts, four forms each, xlsx -> JSONL.

    python scripts/prepare_benign_probe.py --xlsx <CS-Jail-UR_benign_probe_v2_60.xlsx> \
        --provenance "AI-assisted drafting, human-verified by UU"

One row per prompt x form: {id "BN-D1-01::EN", base_id, condition, domain_id, boundary_type, prompt}.
Checks: unique BN-D<n>-<nn> ids matching their domain, all four forms filled, Urdu script in
UR and in no Latin form, no duplicate text within a form. Writes data/benign_probe_v2.jsonl and
.manifest.json (source sha256, counts, provenance as stated by the team).
"""
from __future__ import annotations

import argparse
import collections
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from csjail.utils.io import sha256_file, write_jsonl, write_text_lf  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FORMS = ("EN", "CS", "RU", "UR")
URDU = re.compile(r"[؀-ۿ]")


def main(argv=None) -> int:
    import json

    import openpyxl

    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", required=True)
    ap.add_argument("--provenance", required=True, help="how the prompts were written and checked")
    ap.add_argument("--out", default=str(ROOT / "data" / "benign_probe_v2.jsonl"))
    args = ap.parse_args(argv)
    rows = [r for r in openpyxl.load_workbook(args.xlsx, read_only=True).active.iter_rows(values_only=True)
            if any(v not in (None, "") for v in r)]
    hdr = [str(h).strip() if h is not None else "" for h in rows[0]]
    data = [dict(zip(hdr, r)) for r in rows[1:]]
    problems = []
    ids = [str(d.get("prompt_id") or "") for d in data]
    if len(set(ids)) != len(ids):
        problems.append("duplicate prompt_id")
    out = []
    for d, pid in zip(data, ids):
        if not re.fullmatch(r"BN-(D[1-6])-\d{2}", pid) or pid.split("-")[1] != d.get("domain_id"):
            problems.append(f"{pid}: id/domain mismatch")
        for c in FORMS:
            text = str(d.get(c) or "").strip()
            if not text:
                problems.append(f"{pid}/{c}: empty")
            elif bool(URDU.search(text)) != (c == "UR"):
                problems.append(f"{pid}/{c}: wrong script")
            out.append({"id": f"{pid}::{c}", "base_id": pid, "condition": c, "domain_id": d.get("domain_id"),
                        "boundary_type": d.get("boundary_type"), "prompt": text})
    for c in FORMS:
        texts = [r["prompt"] for r in out if r["condition"] == c]
        if len(set(texts)) != len(texts):
            problems.append(f"{c}: duplicate text")
    if problems:
        print("FAIL:\n  " + "\n  ".join(problems[:20]), file=sys.stderr)
        return 1
    write_jsonl(args.out, out)
    write_text_lf(Path(args.out).with_suffix(".manifest.json"), json.dumps({
        "kind": "benign_probe_manifest", "version": "v2", "source_file": Path(args.xlsx).name,
        "source_sha256": sha256_file(args.xlsx), "n_prompts": len(data), "n_rows": len(out),
        "per_domain": dict(sorted(collections.Counter(d.get("domain_id") for d in data).items())),
        "boundary_types": dict(collections.Counter(d.get("boundary_type") for d in data).most_common()),
        "written_by": sorted({str(d.get("written_by")) for d in data}),
        "translated_by": sorted({str(d.get("translated_by")) for d in data}),
        "provenance": args.provenance, "output_sha256": sha256_file(args.out)}, indent=2) + "\n")
    print(f"[benign] {len(data)} prompts x {len(FORMS)} forms -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
