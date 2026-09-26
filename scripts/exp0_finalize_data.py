#!/usr/bin/env python3
"""Exp 0 — ingest, validate, group and split the FINAL dataset (CPU only).

    python scripts/exp0_finalize_data.py --source-csv data/CS-Jail-UR_final_692.csv

Steps
 1. Convert the seven-column CSV (csjail.convert_final) -- or read an already
    converted JSONL via --dataset.
 2. Structural gate: every family has exactly one EN/CS/RU/UR row, one domain,
    no duplicates, no legacy SM. Counts come from the file, not from code.
 3. QA reports (IDs only, no prompt text): script flags, RU English-clause
    flags, equal variants, exact duplicates, token-Jaccard near-duplicate
    candidates. Nothing is deleted or rewritten.
 4. Duplicate grouping (exact + unresolved/confirmed candidates) -> group_id.
 5. Optional QA ledger import (--qa-ledger) and descriptive agreement
    (--independent-annotations). There is NO kappa gate: agreement is reported
    when independent annotations exist, never stubbed.
 6. Frozen, group-level, domain-stratified split manifest (fresh, or an
    append-only extension of --extend-split).
 7. Everything is written to a temp dir and only moved to
    outputs/exp0/<dataset_version>/ (+ LATEST.json) if every gate passes.
    Failures leave outputs/exp0/FAILED-*/ marked DIAGNOSTIC_ONLY.

On another machine with a committed finalized dir (the dataset JSONL itself
is gitignored), recreate it with the same command plus --restore; it is
copied in only if its sha256 and split_id match the committed manifest.
Files are written with LF line endings so hashes match across OSes.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from csjail.artifacts import EXP0_ROOT, FINALIZED_MARKER, ROOT, file_sha256_or_none  # noqa: E402
from csjail.convert_final import convert, flag_counts  # noqa: E402
from csjail.data import (  # noqa: E402
    SCHEMA_VERSION, DatasetError, Prompt, check_structure, load_dataset, summarize,
)
from csjail.qa import (  # noqa: E402
    content_hash, equal_variant_flags, exact_duplicate_groups, ru_english_flags,
    ru_loanword_inventory, script_flags, token_jaccard_candidates,
)
from csjail.splits import (  # noqa: E402
    attach_heuristic_features, build_groups, dataset_stats, extend_split,
    load_duplicate_decisions, load_manifest, make_splits, pairwise_agreement, verify_split,
)
from csjail.utils.io import sha256_file, write_jsonl, write_text_lf  # noqa: E402

QA_LEDGER_FIELDS = ["family_id", "dataset_version", "reviewer", "review_date",
                    "harmful_eligible", "semantic_equivalence", "condition_validity",
                    "cs_naturalness", "duplicate_decision", "resolution", "status", "notes"]


def _read_ledger(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    missing = [c for c in ("family_id", "status") if rows and c not in rows[0]]
    if missing:
        raise ValueError(f"QA ledger {path} missing columns {missing}")
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--source-csv", help="final seven-column CSV")
    src.add_argument("--dataset", help="already-converted long JSONL")
    ap.add_argument("--eval-size", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scheme", choices=["main", "domain_holdout"], default="main")
    ap.add_argument("--holdout-domains", nargs="*", default=None,
                    help="scheme=domain_holdout only: explicitly named domain(s)")
    ap.add_argument("--extend-split", default=None,
                    help="frozen split_manifest.json to extend append-only")
    ap.add_argument("--near-dup-threshold", type=float, default=0.92,
                    help="token-Jaccard threshold (NOT cosine)")
    ap.add_argument("--duplicate-decisions", default="data/qa/duplicate_decisions.csv")
    ap.add_argument("--qa-ledger", default=None, help="optional QA ledger CSV")
    ap.add_argument("--independent-annotations", default=None,
                    help="optional CSV with rater1_*/rater2_* columns (pre-adjudication)")
    ap.add_argument("--out-root", default=str(EXP0_ROOT))
    ap.add_argument("--overwrite", action="store_true",
                    help="replace an existing finalized dir for the same version")
    ap.add_argument("--restore", action="store_true",
                    help="regenerate the gitignored dataset file into an existing (committed) "
                         "finalized dir; succeeds only if its sha256 and split_id match")
    ap.add_argument("--no-latest", action="store_true",
                    help="do not update LATEST.json (e.g. fixtures/smoke runs)")
    args = ap.parse_args(argv)

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tmp = out_root / f".tmp-{stamp}-{os.getpid()}"
    tmp.mkdir()
    gates: dict[str, str] = {}
    report: dict = {"created_utc": stamp, "schema_version": SCHEMA_VERSION}

    def fail(msg: str, code: int) -> int:
        report["gates"] = gates
        report["failure"] = msg
        write_text_lf(tmp / "DIAGNOSTIC_ONLY", msg + "\n")
        write_text_lf(tmp / "exp0_report.json", json.dumps(report, ensure_ascii=False, indent=2))
        dest = out_root / f"FAILED-{stamp}"
        tmp.rename(dest)
        print(f"[exp0] GATE FAIL: {msg}\n[exp0] diagnostics -> {dest}", file=sys.stderr)
        return code

    # 1. ingest
    try:
        if args.source_csv:
            long_rows, conv = convert(args.source_csv)
            report["conversion"] = {k: v for k, v in conv.items() if k != "review_flags"}
            source_sha = conv["input_sha256"]
            rows = [Prompt.model_validate(r) for r in long_rows]
        else:
            rows = load_dataset(args.dataset)
            source_sha = sha256_file(args.dataset)
    except (ValueError, DatasetError) as e:
        gates["ingest"] = "FAIL"
        return fail(f"ingest: {e}", 1)
    gates["ingest"] = "PASS"

    # 2. exact structural gate
    st = check_structure(rows)
    report["structure"] = {"n_families": st.n_families, "n_rows": st.n_rows,
                           "n_complete": st.n_complete, "errors": st.errors[:50]}
    if not st.ok:
        gates["structure"] = "FAIL"
        return fail(f"structure: {len(st.errors)} errors, e.g. {st.errors[:3]}", 2)
    gates["structure"] = "PASS"
    versions = {r.dataset_version for r in rows}
    chash = content_hash(rows)
    dataset_version = versions.pop() if len(versions) == 1 and None not in versions \
        else f"final-{st.n_families}-{chash[:12]}"

    # 3. QA reports (ids only)
    exact = exact_duplicate_groups(rows)
    cands = token_jaccard_candidates(rows, threshold=args.near_dup_threshold)
    flags = {
        "equal_variants_within_family": equal_variant_flags(rows),
        "exact_duplicates_across_families": exact,
        "script": script_flags(rows),
        "ru_possible_english_clause": ru_english_flags(rows),
        "ru_loanword_inventory": ru_loanword_inventory(rows),
    }
    write_text_lf(tmp / "qa_review_flags.json", json.dumps(flags, ensure_ascii=False, indent=2))
    decisions = load_duplicate_decisions(args.duplicate_decisions)
    with (tmp / "duplicate_candidates.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["family_a", "family_b", "condition", "method",
                                          "score", "threshold", "decision"],
                           lineterminator="\n")
        w.writeheader()
        for c in cands:
            w.writerow({**c, "decision": decisions.get(tuple(sorted((c["family_a"],
                                                                      c["family_b"]))), "")})
    report["qa_flag_counts"] = flag_counts({"review_flags": flags})
    report["qa_flag_counts"]["ru_items_with_technical_loans"] = \
        flags["ru_loanword_inventory"]["n_ru_items_with_technical_loans"]

    # 4. groups
    fam_to_group, group_report = build_groups(
        rows, exact_duplicates=exact, near_duplicate_candidates=cands, decisions=decisions)
    report["groups"] = group_report

    # 5. optional QA ledger / agreement
    qa_summary: dict = {"qa_ledger": None, "agreement": None}
    if args.qa_ledger:
        ledger = _read_ledger(Path(args.qa_ledger))
        unknown = sorted({r["family_id"] for r in ledger} - {r.base_id for r in rows})
        qa_summary["qa_ledger"] = {
            "path": args.qa_ledger, "sha256": sha256_file(args.qa_ledger),
            "n_records": len(ledger), "status_counts": dict(Counter(r["status"] for r in ledger)),
            "unknown_family_ids": unknown[:50],
            "harmful_eligible_no": sorted(r["family_id"] for r in ledger
                                          if (r.get("harmful_eligible") or "").lower() == "no"),
        }
    if args.independent_annotations:
        qa_summary["agreement"] = pairwise_agreement(args.independent_annotations)
    write_text_lf(tmp / "qa_summary.json", json.dumps(qa_summary, ensure_ascii=False, indent=2))

    # 6. split
    try:
        if args.extend_split:
            split = extend_split(load_manifest(args.extend_split), rows,
                                 fam_to_group=fam_to_group, dataset_version=dataset_version)
        else:
            split = make_splits(rows, fam_to_group=fam_to_group, eval_size=args.eval_size,
                                seed=args.seed, scheme=args.scheme,
                                holdout_domains=args.holdout_domains,
                                dataset_version=dataset_version)
    except ValueError as e:
        gates["split"] = "FAIL"
        return fail(f"split: {e}", 3)
    rows = [r.model_copy(update={"group_id": fam_to_group[r.base_id],
                                 "dataset_version": dataset_version}) for r in rows]
    rows = attach_heuristic_features(rows)
    errs = verify_split(split, rows)
    if errs:
        gates["split"] = "FAIL"
        return fail(f"split verification: {errs[:3]}", 3)
    gates["split"] = "PASS"
    write_text_lf(tmp / "split_manifest.json", json.dumps(split, ensure_ascii=False, indent=2))

    # 7. dataset + manifest
    ds_name = "dataset_final.jsonl"
    write_jsonl(tmp / ds_name, [r.model_dump() for r in rows])
    write_text_lf(tmp / "dataset_stats.json",
        json.dumps({"summary": summarize(rows), "stats": dataset_stats(rows)},
                   ensure_ascii=False, indent=2))
    manifest = {
        "dataset_version": dataset_version,
        "schema_version": SCHEMA_VERSION,
        "source_sha256": source_sha,
        "source_path": args.source_csv or args.dataset,
        "normalized_content_sha256": chash,
        "dataset_file": ds_name,
        "dataset_file_sha256": sha256_file(tmp / ds_name),
        "split_file": "split_manifest.json",
        "split_id": split["meta"]["split_id"],
        "split_scheme": split["meta"]["scheme"],
        "qa_ledger_sha256": file_sha256_or_none(args.qa_ledger),
        "duplicate_decisions_sha256": file_sha256_or_none(args.duplicate_decisions),
        "domain_dictionary_version": (report.get("conversion") or {}).get("domain_dictionary_version"),
        "n_families": st.n_families, "n_rows": st.n_rows,
        "gates": gates,
        "notes": ["no dataset-level kappa gate (docs/PROTOCOL.md §3.3)",
                  "cmi/urdu_word_ratio present only as *_heuristic (unvalidated)"],
    }
    write_text_lf(tmp / "dataset_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    report["gates"] = gates
    write_text_lf(tmp / "exp0_report.json", json.dumps(report, ensure_ascii=False, indent=2))
    write_text_lf(tmp / FINALIZED_MARKER, stamp + "\n")

    dest = out_root / dataset_version
    if dest.exists() and args.restore:
        old = json.loads((dest / "dataset_manifest.json").read_text(encoding="utf-8"))
        same = (old["dataset_file_sha256"] == manifest["dataset_file_sha256"]
                and old["split_id"] == manifest["split_id"])
        if same:
            shutil.copyfile(tmp / ds_name, dest / ds_name)
        shutil.rmtree(tmp)
        if not same:
            print(f"[exp0] RESTORE FAIL: regenerated dataset/split do not match {dest} "
                  f"(sha {manifest['dataset_file_sha256'][:12]} vs {old['dataset_file_sha256'][:12]}, "
                  f"split {manifest['split_id']} vs {old['split_id']})", file=sys.stderr)
            return 5
        print(f"[exp0] restored {ds_name} into {dest} (sha256 and split_id verified)")
        return 0
    if dest.exists():
        if not args.overwrite:
            shutil.rmtree(tmp)
            print(f"[exp0] {dest} already finalized; refusing to overwrite a frozen "
                  "artifact (pass --overwrite to replace it deliberately)", file=sys.stderr)
            return 4
        shutil.rmtree(dest)
    tmp.rename(dest)
    if not args.no_latest:
        rel = os.path.relpath(dest, ROOT).replace("\\", "/")
        write_text_lf(out_root / "LATEST.json", json.dumps({"dir": rel}, indent=2) + "\n")

    m = split["meta"]
    print(f"[exp0] dataset_version: {dataset_version}")
    print(f"[exp0] families: {st.n_families}  rows: {st.n_rows}  (structure PASS)")
    print(f"[exp0] split {m['split_id']} scheme={m['scheme']}: "
          + ", ".join(f"{s}={c['total']}" for s, c in m["counts"].items()))
    print(f"[exp0] groups: {group_report}")
    print(f"[exp0] QA flags (review, not deletions): {report['qa_flag_counts']}")
    print(f"[exp0] FINALIZED -> {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
