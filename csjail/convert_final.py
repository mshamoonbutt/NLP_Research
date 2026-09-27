"""Convert a CS-Jail-UR release CSV to the long JSONL schema.

Required columns (any order): prompt_id, domain_id, domain_name, EN, CS, RU, UR.
Any additional columns (the 791 release adds row_number, evaluation_stratum,
approval_status) are PRESERVED per family in `provenance.source_metadata`,
never dropped. `row_number` is display/order metadata only -- never an ID.
`evaluation_stratum` must be `harmful` (it describes the prompt population,
not an observed harmful response) and `approval_status` must be `approved`
(supplied review metadata, not evidence of agreement or judge accuracy);
any other value FAILS conversion rather than being silently included.
Output: one row per (family, condition) -- exactly four per family.

Rules
- `prompt_id` is an opaque STRING (never int-cast). It becomes `base_id`
  unchanged; the row id is `csjail.data.make_row_id(base_id, condition)`.
  Domain is NOT embedded in any ID, so a domain correction cannot change a
  family's identity.
- (domain_id, domain_name) must match the versioned dictionary in
  configs/domains.yaml. `harm_category` is a compatibility alias == domain_id.
- Text is preserved. The ONLY normalization is removing leading/trailing
  whitespace and converting CRLF/CR line endings to LF; whenever that changes
  a cell, the original is kept in `provenance.source_text` and the exact
  transformation in `provenance.text_normalization`. Internal newlines survive.
- Absent metadata (author, severity, authenticity, style, model assistance)
  stays absent/null. Nothing is invented.
- Nothing is deleted, renumbered or rebalanced. Blank cells, duplicate IDs,
  an unknown domain or a wrong column set FAIL the conversion; questionable
  content is REPORTED (see csjail.qa) for human review.

Usage:
    python -m csjail.convert_final --input data/CS-Jail-UR_final_approved_791.csv \\
        --out data/csjail_final.jsonl --report outputs/exp0/convert_report.json
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from csjail.data import CONDITIONS, ROW_ID_SEP, Prompt, make_row_id, validate_structure
from csjail.qa import (
    content_hash,
    equal_variant_flags,
    exact_duplicate_groups,
    ru_english_flags,
    script_flags,
)
from csjail.utils.io import sha256_file, write_jsonl

ROOT = Path(__file__).resolve().parent.parent
DOMAINS_CFG = ROOT / "configs" / "domains.yaml"
REQUIRED_COLUMNS = ["prompt_id", "domain_id", "domain_name", "EN", "CS", "RU", "UR"]
ALLOWED_METADATA_VALUES = {"evaluation_stratum": {"harmful"}, "approval_status": {"approved"}}
CONVERTER_VERSION = "convert_final-v2"


def load_domain_dictionary(path: str | Path = DOMAINS_CFG) -> tuple[str, dict[str, str]]:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return cfg["version"], {str(k): str(v) for k, v in cfg["domains"].items()}


def id_namespace(pid: str) -> tuple[str, str | None]:
    """Descriptive namespace + lineage hint derived ONLY from the ID string.
    The pattern shows that batches differ; it does not establish authorship
    or unchanged text."""
    if re.fullmatch(r"\d+", pid):
        return "numeric", None
    m = re.fullmatch(r"CSJUR-R-(\d+)-(\d+)", pid)
    if m:
        return "CSJUR-R", f"id-references-replaced-prompt:{m.group(1)}"
    m = re.fullmatch(r"(CSJUR-[A-Z]+\d*)-\d+", pid)
    if m:
        return m.group(1), None
    return "other", None


def _normalize(text: str) -> tuple[str, str | None]:
    steps = []
    out = text
    if "\r" in out:
        out = out.replace("\r\n", "\n").replace("\r", "\n")
        steps.append("crlf_to_lf")
    stripped = out.strip()
    if stripped != out:
        steps.append("strip_outer_whitespace")
        out = stripped
    return out, ("+".join(steps) if steps else None)


def convert(input_path: str | Path, *, domains_path: str | Path = DOMAINS_CFG,
            ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = Path(input_path)
    dom_version, domains = load_domain_dictionary(domains_path)
    src_sha = sha256_file(path)

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        columns = list(reader.fieldnames or [])
        missing = [c for c in REQUIRED_COLUMNS if c not in columns]
        if missing or len(set(columns)) != len(columns):
            raise ValueError(f"required columns {REQUIRED_COLUMNS} (missing {missing}); "
                             f"got {columns}")
        src_rows = list(reader)
    extra_columns = [c for c in columns if c not in REQUIRED_COLUMNS]

    errors: list[str] = []
    seen: Counter[str] = Counter(r["prompt_id"] for r in src_rows)
    for pid, n in seen.items():
        if n > 1:
            errors.append(f"duplicate prompt_id {pid!r} ({n} rows)")

    out: list[dict[str, Any]] = []
    for i, src in enumerate(src_rows, start=1):
        pid = src["prompt_id"]
        if pid is None or pid.strip() == "":
            errors.append(f"row {i}: blank prompt_id")
            continue
        if pid != pid.strip():
            errors.append(f"row {i}: prompt_id {pid!r} has outer whitespace")
            continue
        if ROW_ID_SEP in pid:
            errors.append(f"row {i}: prompt_id {pid!r} contains reserved {ROW_ID_SEP!r}")
            continue
        did, dname = (src["domain_id"] or "").strip(), (src["domain_name"] or "").strip()
        if did not in domains:
            errors.append(f"row {i} ({pid}): unknown domain_id {did!r}")
            continue
        if domains[did] != dname:
            errors.append(f"row {i} ({pid}): domain_name {dname!r} != dictionary "
                          f"{domains[did]!r} for {did}")
            continue
        meta = {c: (src[c] or "").strip() for c in extra_columns}
        bad = [f"{c}={v!r}" for c, v in meta.items()
               if c in ALLOWED_METADATA_VALUES and v not in ALLOWED_METADATA_VALUES[c]]
        if bad:
            errors.append(f"row {i} ({pid}): unsupported {', '.join(bad)}")
            continue
        ns, hint = id_namespace(pid)
        for cond in CONDITIONS:
            raw = src[cond]
            if raw is None or raw.strip() == "":
                errors.append(f"row {i} ({pid}): blank {cond} text")
                continue
            text, norm = _normalize(raw)
            out.append({
                "id": make_row_id(pid, cond),
                "base_id": pid,
                "condition": cond,
                "prompt": text,
                "harm_category": did,
                "domain_id": did,
                "domain_name": dname,
                "provenance": {
                    "source_file": path.name,
                    "source_sha256": src_sha,
                    "source_row": i,
                    "source_column": cond,
                    "id_namespace": ns,
                    "lineage_hint": hint,
                    "author": None,
                    "model_assistance": None,
                    "text_normalization": norm,
                    "source_text": raw if norm else None,
                    "source_metadata": meta or None,
                },
            })
    if errors:
        joined = "\n  ".join(errors[:20])
        raise ValueError(f"conversion failed ({len(errors)} errors):\n  {joined}")

    prompts = [Prompt.model_validate(r) for r in out]
    validate_structure(prompts)
    chash = content_hash(prompts)
    n_fam = len(src_rows)
    version = f"final-{n_fam}-{chash[:12]}"
    for r in out:
        r["dataset_version"] = version

    report = {
        "converter_version": CONVERTER_VERSION,
        "input_path": str(path),
        "input_sha256": src_sha,
        "normalized_content_sha256": chash,
        "dataset_version": version,
        "domain_dictionary_version": dom_version,
        "n_families": n_fam,
        "n_long_rows": len(out),
        "families_by_domain": dict(sorted(Counter(r["domain_id"] for r in src_rows).items())),
        "families_by_id_namespace": dict(sorted(Counter(id_namespace(r["prompt_id"])[0]
                                                        for r in src_rows).items())),
        "n_cells_normalized": sum(1 for r in out if r["provenance"]["text_normalization"]),
        "normalizations": dict(Counter(r["provenance"]["text_normalization"]
                                       for r in out if r["provenance"]["text_normalization"])),
        "n_cells_with_internal_newlines": sum(1 for r in out if "\n" in r["prompt"]),
        "extra_columns_preserved": extra_columns,
        "metadata_value_counts": {c: dict(Counter((r[c] or "").strip() for r in src_rows))
                                  for c in extra_columns if c in ALLOWED_METADATA_VALUES},
        "review_flags": {
            "equal_variants_within_family": equal_variant_flags(prompts),
            "exact_duplicates_across_families": exact_duplicate_groups(prompts),
            "script": script_flags(prompts),
            "ru_possible_english_clause": ru_english_flags(prompts),
        },
    }
    return out, report


def flag_counts(report: dict[str, Any]) -> dict[str, int]:
    rf = report["review_flags"]
    return {
        "equal_variants_within_family": len(rf["equal_variants_within_family"]),
        "exact_duplicates_across_families": len(rf["exact_duplicates_across_families"]),
        "script": dict(Counter(f["flag"] for f in rf["script"])),
        "ru_possible_english_clause": len(rf["ru_possible_english_clause"]),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="final seven-column CSV")
    ap.add_argument("--out", required=True, help="output long JSONL")
    ap.add_argument("--report", default=None, help="optional JSON report path")
    args = ap.parse_args(argv)

    if not Path(args.input).exists():
        print(f"FAIL: input not found: {args.input}", file=sys.stderr)
        return 1
    try:
        rows, report = convert(args.input)
    except ValueError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 2
    write_jsonl(args.out, rows)
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
    summary = {k: v for k, v in report.items() if k != "review_flags"}
    summary["review_flag_counts"] = flag_counts(report)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[convert_final] wrote {len(rows)} rows ({report['n_families']} families) -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
